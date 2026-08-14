"""AI 스마트 믹싱용 현장 이미지 분석 API.

현재는 정상 이미지가 충분히 쌓이기 전까지 기준 이미지 기반 시범 판정을 제공합니다.
정상 이미지가 30장 이상 쌓이면 /train-patchcore를 호출해 PatchCore 모델을 학습하도록
확장할 수 있습니다. 서버 관리자 키나 Supabase 비밀 키는 이 서비스에 저장하지 않습니다.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import shutil
import uuid
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parent
NORMAL_DIR = APP_ROOT / "data" / "normal"
MODEL_DIR = APP_ROOT / "models"
MIXING_MODEL_PATH = MODEL_DIR / "xgboost_mixing.pkl"
MIXING_TRAINING_PATH = APP_ROOT / "data" / "training" / "mixing_records.json"
LEARNING_DIR = APP_ROOT / "data" / "learning"
LEARNING_RECORDS = LEARNING_DIR / "records.json"
NORMAL_NAMES = (
    "learning-ladle-wall.jpg",
    "learning-ladle-bottom.jpg",
    "learning-tundish-cover.jpg",
    "learning-burner.jpg",
)
SITE_FILE = "ai-smart-mixing-integrated.html"
SITE_ASSETS = {SITE_FILE, *NORMAL_NAMES}

MIXING_FEATURES = ("temperature_c", "humidity_pct", "amount_kg", "cycle_no", "round_no", "material_code")
MIXING_TARGETS = ("water_l", "mixing_min", "hopper_wait_min", "risk_pct")
MATERIAL_CODES = {"래들 벽체": 1, "래들 바닥": 2, "턴디쉬 카바": 3, "저시멘트 캐스터블": 4}

app = FastAPI(title="AI 스마트 믹싱 이미지 분석 API", version="0.1.0")
MODEL_SCHEMA_VERSION = "literature_informed_synthetic_v2"
MATERIAL_PROFILES = {
    1: {"name": "ladle_wall", "water_pct": 5.2, "mixing_min": 5.0},
    2: {"name": "ladle_bottom", "water_pct": 5.0, "mixing_min": 5.0},
    3: {"name": "tundish_cover", "water_pct": 4.8, "mixing_min": 4.0},
    4: {"name": "low_cement", "water_pct": 5.5, "mixing_min": 5.0},
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8990", "http://localhost:8990", "http://127.0.0.1:8765", "http://localhost:8765"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _material_code(value: object) -> int:
    text = str(value or "")
    if text in MATERIAL_CODES:
        return MATERIAL_CODES[text]
    return 0


def _feature_row(row: dict) -> list[float]:
    return [
        float(row.get("temperature_c", 25)),
        float(row.get("humidity_pct", 60)),
        float(row.get("amount_kg", 500)),
        float(row.get("cycle_no", 1)),
        float(row.get("round_no", 1)),
        float(row.get("material_code", _material_code(row.get("material_type")))),
    ]


def _literature_rule(temperature: float, humidity: float, amount: float, cycle: int, round_no: int, material_code: int) -> dict:
    """Published castable installation guidance translated into demo rules.

    16-29C is the preferred installation range. High temperature accelerates hydration
    and shortens working time. Product water remains within its own specified range.
    Humidity and queue sequence are intentionally small operational assumptions until
    field measurements replace this literature-informed synthetic dataset.
    """
    profile = MATERIAL_PROFILES.get(material_code, MATERIAL_PROFILES[4])
    hot_penalty = max(0.0, temperature - 29.0) * 1.35
    cold_penalty = max(0.0, 16.0 - temperature) * 0.55
    humidity_penalty = max(0.0, humidity - 75.0) * 0.12
    queue_penalty = max(0, round_no - 1) * 1.1 + max(0, cycle - 1) * 0.45
    risk = max(2.0, min(30.0, 4.0 + hot_penalty + cold_penalty + humidity_penalty + queue_penalty))
    water_pct = profile["water_pct"] + (0.08 if temperature > 29 else 0.0) - (0.05 if humidity > 80 else 0.0)
    return {
        "water_l": round(amount * water_pct / 100.0, 2),
        "mixing_min": profile["mixing_min"],
        "hopper_wait_min": round(max(0.0, min(20.0, 20.0 - risk * 0.72)), 2),
        "risk_pct": round(risk, 2),
    }


def _seed_training_rows() -> list[dict]:
    """발표용 초기 모델을 위한 명시적 기준 데이터. 실제 운영 전 현장 라벨 데이터로 교체한다."""
    rows = []
    temperatures = (10, 16, 20, 25, 29, 32, 36, 40)
    humidities = (40, 55, 65, 75, 85)
    for material_code in MATERIAL_PROFILES:
        for cycle in range(1, 4):
            for round_no in range(1, 4):
                for index, temperature in enumerate(temperatures):
                    humidity = humidities[(index + cycle + round_no + material_code) % len(humidities)]
                    amount = 500 if round_no < 3 else 250
                    rows.append({
                        "temperature_c": temperature,
                        "humidity_pct": humidity,
                        "amount_kg": amount,
                        "cycle_no": cycle,
                        "round_no": round_no,
                        "material_code": material_code,
                        **_literature_rule(temperature, humidity, amount, cycle, round_no, material_code),
                    })
    return rows


def _fit_mixing_model(rows: list[dict], source: str) -> dict:
    if len(rows) < 8:
        raise HTTPException(status_code=422, detail="XGBoost 학습에는 최소 8건의 라벨 작업 데이터가 필요합니다.")
    X = np.asarray([_feature_row(row) for row in rows], dtype=np.float32)
    models = {}
    validation = {}
    train_idx, test_idx = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42)
    for target in MIXING_TARGETS:
        y = np.asarray([float(row[target]) for row in rows], dtype=np.float32)
        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X, y, verbose=False)
        models[target] = model
        holdout_model = XGBRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror", random_state=42,
        )
        holdout_model.fit(X[train_idx], y[train_idx], verbose=False)
        predicted = holdout_model.predict(X[test_idx])
        scale = max(float(np.ptp(y)), 1.0)
        mae = float(mean_absolute_error(y[test_idx], predicted))
        validation[target] = {"mae": round(mae, 3), "score": round(max(0.0, min(1.0, 1.0 - mae / scale)), 3)}
    validation_score = float(np.mean([item["score"] for item in validation.values()]))
    feature_min = X.min(axis=0).tolist()
    feature_max = X.max(axis=0).tolist()
    artifact = {"models": models, "features": MIXING_FEATURES, "rows": len(rows), "source": source, "schema_version": MODEL_SCHEMA_VERSION, "trained_at": datetime.now(timezone.utc).isoformat(), "validation": validation, "validation_score": round(validation_score, 3), "feature_min": feature_min, "feature_max": feature_max}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MIXING_MODEL_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)
    MIXING_TRAINING_PATH.parent.mkdir(parents=True, exist_ok=True)
    MIXING_TRAINING_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def _ensure_mixing_model() -> dict:
    if MIXING_MODEL_PATH.exists():
        try:
            with MIXING_MODEL_PATH.open("rb") as handle:
                artifact = pickle.load(handle)
                if "validation" in artifact and "feature_min" in artifact and "feature_max" in artifact and (artifact.get("source") == "field_worklogs" or artifact.get("schema_version") == MODEL_SCHEMA_VERSION):
                    return artifact
        except Exception:
            pass
    rows = _seed_training_rows()
    return _fit_mixing_model(rows, "literature_informed_synthetic")


def _recommendation_confidence(artifact: dict, features: np.ndarray) -> dict:
    """검증 성능과 현재 입력값이 학습 범위 안에 있는지를 합쳐 계산한다."""
    validation_score = float(artifact.get("validation_score", 0.0))
    mins = np.asarray(artifact.get("feature_min", []), dtype=np.float32)
    maxs = np.asarray(artifact.get("feature_max", []), dtype=np.float32)
    if mins.size != features.shape[1] or maxs.size != features.shape[1]:
        coverage = 0.5
    else:
        span = np.maximum(maxs - mins, 1.0)
        below = np.maximum(mins - features[0], 0.0) / span
        above = np.maximum(features[0] - maxs, 0.0) / span
        coverage = float(max(0.0, min(1.0, 1.0 - np.mean(below + above))))
    confidence = round((validation_score * 0.7 + coverage * 0.3) * 100, 1)
    return {"confidence_pct": confidence, "validation_score": round(validation_score * 100, 1), "condition_coverage_pct": round(coverage * 100, 1)}


def prepare_normal_images() -> int:
    """기존 사이트의 정상 참고 이미지를 학습 폴더로 준비한다."""
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    for image_name in NORMAL_NAMES:
        source = WORKSPACE_ROOT / image_name
        target = NORMAL_DIR / image_name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    return len(list(NORMAL_DIR.glob("*.*")))


def inspect_visual_features(raw: bytes) -> dict[str, float]:
    """PatchCore 학습 전 사용되는 실제 영상 특징 기반의 시범 검사."""
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽지 못했습니다.") from exc

    rgb = np.asarray(image.resize((512, 512)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    dark_ratio = float(np.mean(gray < 65))
    bright_ratio = float(np.mean(gray > 220))
    laplacian_std = float(cv2.Laplacian(gray, cv2.CV_64F).std())

    # 균열은 긴 경계·어두운 선형 패턴이 많을수록 높게, 급속 경화는 표면 질감 변화가 클수록 높게 본다.
    crack_score = min(30.0, round(3.0 + edge_density * 230 + dark_ratio * 35, 1))
    hardening_score = min(30.0, round(4.0 + edge_density * 150 + max(0, laplacian_std - 35) * 0.12 + bright_ratio * 10, 1))
    return {
        "hardening_score": hardening_score,
        "crack_score": crack_score,
        "edge_density": round(edge_density, 4),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "texture_variation": round(laplacian_std, 2),
    }


def improvement_note(hardening: float, crack: float) -> str:
    if hardening > 10 or crack > 10:
        return "표면 상태 변화가 감지되었습니다. 해당 차수의 호퍼 대기시간, 수분량, 지연제 사용 여부를 작업일지와 대조해 확인하세요."
    return "뚜렷한 급속 경화·균열 의심 신호는 낮습니다. 같은 촬영 거리와 조명으로 사진을 추가 확보해 모델 기준을 보강하세요."


def classify_image_scope(raw: bytes) -> dict[str, object]:
    """정상 내화물 참고 이미지와의 시각 특징 유사도로 분석 대상 여부를 먼저 확인한다."""
    prepare_normal_images()
    target = inspect_visual_features(raw)
    references = []
    for path in NORMAL_DIR.glob("*.*"):
        try:
            references.append(inspect_visual_features(path.read_bytes()))
        except Exception:
            continue
    if not references:
        return {"image_scope": True, "scope_confidence": 0.5, "scope_message": "참고 이미지가 없어 분석 대상 확인을 보류했습니다."}

    # 밝은 배경이 넓고 질감 경계가 적은 문서·홍보물·화면 캡처는
    # 내화물 작업면과 특징이 일부 겹칠 수 있으므로 분석 대상에서 제외한다.
    # 정상 참고 내화물 이미지는 어두운 표면 질감과 높은 경계 밀도를 가진다.
    document_like = target["bright_ratio"] >= 0.45 and target["edge_density"] <= 0.14
    if document_like:
        return {
            "image_scope": False,
            "scope_confidence": 0.05,
            "scope_message": "문서·홍보물·화면 캡처 형태로 보여 내화물 작업면 분석을 중단했습니다.",
        }

    keys = ("edge_density", "dark_ratio", "bright_ratio", "texture_variation")
    scales = {"edge_density": 0.08, "dark_ratio": 0.35, "bright_ratio": 0.35, "texture_variation": 70.0}
    distances = []
    for reference in references:
        distance = sum(abs(float(target[key]) - float(reference[key])) / scales[key] for key in keys) / len(keys)
        distances.append(distance)
    confidence = round(max(0.0, min(1.0, 1.0 - min(distances))), 2)
    return {
        "image_scope": confidence >= 0.38,
        "scope_confidence": confidence,
        "scope_message": "내화물 현장 이미지 특징과 비교했습니다." if confidence >= 0.38 else "내화물 현장 사진 특징과 충분히 일치하지 않아 분석을 중단했습니다.",
    }


@app.get("/health")
def health() -> dict:
    sample_count = prepare_normal_images()
    return {
        "status": "ok",
        "mode": "운영 시스템 연동 대기",
        "analysis_mode": "기준 이미지 기반 시범 판정",
        "normal_image_count": sample_count,
        "patchcore_ready": sample_count >= 30 and (MODEL_DIR / "patchcore").exists(),
        "capabilities": {
            "work_order_storage": "supabase",
            "cycle_storage": "supabase",
            "image_storage": "supabase_storage",
            "ai_training": "not_started",
            "plc_connection": "not_connected",
        },
    }


@app.get("/model/status")
def mixing_model_status() -> dict:
    artifact = _ensure_mixing_model()
    return {
        "status": "ready",
        "model": "XGBoost",
        "model_version": "XGBoost v2.4",
        "trained_at": artifact["trained_at"],
        "training_rows": artifact["rows"],
        "training_source": artifact["source"],
        "dataset_basis": "literature_informed_synthetic" if artifact["source"] != "field_worklogs" else "field_worklogs",
        "validation_score_pct": round(float(artifact.get("validation_score", 0.0)) * 100, 1),
        "validation": artifact.get("validation", {}),
        "targets": list(MIXING_TARGETS),
    }


@app.post("/model/recommend")
def recommend_mixing(payload: dict) -> dict:
    """현재 환경·배합 조건을 실제 XGBoost 모델에 넣어 추천값을 반환한다."""
    artifact = _ensure_mixing_model()
    features = np.asarray([_feature_row(payload)], dtype=np.float32)
    predictions = {name: float(model.predict(features)[0]) for name, model in artifact["models"].items()}
    risk = round(max(0.0, min(30.0, predictions["risk_pct"])), 1)
    water = round(max(0.0, predictions["water_l"]), 1)
    mixing = round(max(1.0, predictions["mixing_min"]), 1)
    hopper_wait = round(max(0.0, predictions["hopper_wait_min"]), 1)
    confidence = _recommendation_confidence(artifact, features)
    return {
        "model": "XGBoost",
        "model_version": "XGBoost v2.4",
        "trained_at": artifact["trained_at"],
        "training_rows": artifact["rows"],
        "training_source": artifact["source"],
        "dataset_basis": "literature_informed_synthetic" if artifact["source"] != "field_worklogs" else "field_worklogs",
        **confidence,
        "water_l": water,
        "mixing_min": mixing,
        "hopper_wait_min": hopper_wait,
        "risk_pct": risk,
        "retarder": risk > 10,
        "explanation": "학습된 작업 데이터에서 현재 온도·습도·내화물·회차 조건과 가장 유사한 패턴을 기준으로 계산했습니다.",
    }


@app.post("/model/fit")
def fit_mixing_model(payload: dict) -> dict:
    """검증된 작업일지 데이터를 받아 XGBoost 모델을 실제로 다시 학습한다."""
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="rows 배열이 필요합니다.")
    artifact = _fit_mixing_model(rows, "field_worklogs")
    return {
        "status": "trained",
        "model": "XGBoost",
        "model_version": "XGBoost v2.4",
        "training_rows": artifact["rows"],
        "trained_at": artifact["trained_at"],
        "validation_score_pct": round(float(artifact.get("validation_score", 0.0)) * 100, 1),
        "validation": artifact.get("validation", {}),
        "message": "검증된 작업 데이터로 모델을 다시 학습했습니다.",
    }


@app.get("/system/status")
def system_status() -> dict:
    """사이트가 운영 모드에서 어떤 기능을 사용할 수 있는지 표시한다."""
    sample_count = prepare_normal_images()
    return {
        "system_status": "ready",
        "message": "작업 데이터 저장과 AI 분석 연결 구조가 준비되었습니다.",
        "database": "supabase 연결은 사이트에서 수행",
        "storage": "refractory-images 버킷 사용",
        "analysis": "학습 전 시범 판정",
        "patchcore": "학습 이미지와 모델 배포 후 활성화",
        "normal_image_count": sample_count,
    }


@app.get("/demo-image")
def demo_image():
    """공개 데모 이미지의 브라우저 CORS 차단을 피하기 위한 안전한 프록시입니다."""
    source = "https://upload.wikimedia.org/wikipedia/commons/0/00/Refractory_bricks_lining.jpg"
    try:
        request = Request(source, headers={"User-Agent": "AI-Smart-Mixing-Demo/1.0"})
        with urlopen(request, timeout=12) as response:
            content = response.read()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="공개 데모 이미지를 불러오지 못했습니다.") from exc
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="JPG, PNG, WEBP 이미지 파일만 분석할 수 있습니다.")

    raw = await file.read()
    if len(raw) == 0 or len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="이미지 파일은 1바이트 이상 15MB 이하로 올려주세요.")

    feature = inspect_visual_features(raw)
    scope = classify_image_scope(raw)
    if not scope["image_scope"]:
        return {
            "analysis_status": "분석 대상 아님",
            "analysis_mode": "내화물 현장 사진 1차 확인",
            "model_version": "PatchCore 학습 준비 · 이미지 범위 확인 v0.1",
            "image_scope": False,
            "scope_confidence": scope["scope_confidence"],
            "hardening_score": None,
            "crack_score": None,
            "analysis_result": scope["scope_message"],
            "improvement_note": "래들·턴디쉬·버너 등 부정형 내화물 작업면이 보이는 사진으로 다시 업로드하세요.",
            "feature_summary": feature,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "analysis_status": "시범 판정 완료",
        "analysis_mode": "기준 이미지 기반 시범 판정",
        "model_version": "PatchCore 학습 준비 · Visual baseline v0.1",
        "image_scope": True,
        "scope_confidence": scope["scope_confidence"],
        "hardening_score": feature["hardening_score"],
        "crack_score": feature["crack_score"],
        "analysis_result": "표면 질감·경계 패턴·명암 분포를 기준 이미지와 비교했습니다.",
        "improvement_note": improvement_note(feature["hardening_score"], feature["crack_score"]),
        "feature_summary": feature,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/learning-images")
async def save_learning_image(
    file: UploadFile = File(...),
    job_name: str = Form(""),
    cycle: str = Form(""),
    round_name: str = Form(""),
    analysis: str = Form("{}"),
) -> dict:
    """작업자가 승인한 학습 이미지를 파일과 메타데이터로 함께 저장한다.

    운영 환경에서는 이 저장 지점을 Supabase Storage/image_inspections로 교체할 수 있고,
    로컬 데모에서는 재시작 후에도 확인할 수 있도록 ai-service/data/learning에 저장한다.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 학습 데이터로 저장할 수 있습니다.")
    raw = await file.read()
    if len(raw) == 0 or len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="이미지 파일은 1바이트 이상 15MB 이하로 올려주세요.")
    try:
        analysis_data = json.loads(analysis or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="분석 결과 형식이 올바르지 않습니다.") from exc
    if analysis_data.get("image_scope") is not True:
        raise HTTPException(status_code=409, detail="내화물 작업면으로 확인된 이미지만 학습 반영할 수 있습니다.")

    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    record_id = str(uuid.uuid4())
    suffix = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    image_path = LEARNING_DIR / f"{record_id}{suffix}"
    image_path.write_bytes(raw)
    record = {
        "id": record_id,
        "image_path": str(image_path.relative_to(APP_ROOT)).replace("\\", "/"),
        "original_name": file.filename or "uploaded-image",
        "job_name": job_name,
        "cycle": cycle,
        "round": round_name,
        "analysis": analysis_data,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    records = []
    if LEARNING_RECORDS.exists():
        try:
            records = json.loads(LEARNING_RECORDS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            records = []
    records.append(record)
    LEARNING_RECORDS.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "saved", "record_id": record_id, "image_path": record["image_path"], "saved_at": record["saved_at"]}


@app.delete("/learning-images/{record_id}")
def delete_learning_image(record_id: str) -> dict:
    """승인된 학습 이미지와 메타데이터를 함께 삭제한다."""
    if not LEARNING_RECORDS.exists():
        raise HTTPException(status_code=404, detail="삭제할 학습 데이터를 찾지 못했습니다.")
    try:
        records = json.loads(LEARNING_RECORDS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="학습 데이터 목록을 읽지 못했습니다.") from exc
    target = next((record for record in records if record.get("id") == record_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="삭제할 학습 데이터를 찾지 못했습니다.")
    image_path = APP_ROOT / target.get("image_path", "")
    if image_path.exists():
        image_path.unlink()
    LEARNING_RECORDS.write_text(json.dumps([record for record in records if record.get("id") != record_id], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "deleted", "record_id": record_id}


@app.post("/train-patchcore")
def train_patchcore() -> dict:
    sample_count = prepare_normal_images()
    if sample_count < 30:
        raise HTTPException(
            status_code=409,
            detail=f"PatchCore 학습에는 정상 이미지 30장 이상이 필요합니다. 현재 {sample_count}장입니다.",
        )
    # 실제 운영에서는 GPU 서버에서 Anomalib PatchCore 학습을 실행하고,
    # 생성된 checkpoint를 MODEL_DIR/patchcore에 저장하도록 배치 작업으로 분리한다.
    return {
        "status": "ready",
        "message": "정상 이미지 기준을 충족했습니다. GPU 학습 작업을 시작할 수 있습니다.",
        "normal_image_count": sample_count,
    }


@app.get("/")
def presentation_site() -> FileResponse:
    """발표용 통합 사이트와 분석 API를 하나의 Render 주소에서 제공한다."""
    return FileResponse(WORKSPACE_ROOT / SITE_FILE)


@app.get("/{asset_name}")
def presentation_asset(asset_name: str) -> FileResponse:
    if asset_name not in SITE_ASSETS:
        raise HTTPException(status_code=404, detail="요청한 파일을 찾을 수 없습니다.")
    return FileResponse(WORKSPACE_ROOT / asset_name)

