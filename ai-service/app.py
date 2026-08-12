"""AI 스마트 믹싱용 현장 이미지 분석 API.

현재는 정상 이미지가 충분히 쌓이기 전까지 기준 이미지 기반 시범 판정을 제공합니다.
정상 이미지가 30장 이상 쌓이면 /train-patchcore를 호출해 PatchCore 모델을 학습하도록
확장할 수 있습니다. 서버 관리자 키나 Supabase 비밀 키는 이 서비스에 저장하지 않습니다.
"""

from __future__ import annotations

import io
import json
import shutil
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from xgboost import XGBRegressor

APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parent
NORMAL_DIR = APP_ROOT / "data" / "normal"
MODEL_DIR = APP_ROOT / "models"
TRAINING_META_PATH = APP_ROOT / "data" / "xgboost_training.json"
NORMAL_NAMES = (
    "learning-ladle-wall.jpg",
    "learning-ladle-bottom.jpg",
    "learning-tundish-cover.jpg",
    "learning-burner.jpg",
)
SITE_FILE = "ai-smart-mixing-integrated.html"
SITE_ASSETS = {SITE_FILE, *NORMAL_NAMES}

app = FastAPI(title="AI 스마트 믹싱 이미지 분석 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8990", "http://localhost:8990"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


SENSOR_STATE = {
    "temperature": 31.0,
    "humidity": 78.0,
    "source": "demo-sensor",
    "season": "summer",
    "amount_kg": 500.0,
    "cycle": 1,
    "round": 1,
    "material": "저시멘트 캐스터블",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}


def _material_code(material: str) -> float:
    """내화물 종류를 모델용 범주값으로 변환한다."""
    name = str(material or "")
    if "래들 바닥" in name:
        return 1.0
    if "턴디시" in name or "턴디쉬" in name:
        return 2.0
    if "버너" in name:
        return 3.0
    return 0.0


def build_demo_xgboost_model() -> XGBRegressor:
    """공개 기술자료의 일반 경향을 반영한 발표용 합성 작업 데이터.

    반영 경향:
    - 물량·수분은 배합 일관성과 작업성에 직접 영향
    - 온도·습도·호퍼 대기시간이 길수록 작업 가능 시간이 짧아질 수 있음
    - 믹싱시간은 재료별 최적 범위를 벗어나면 위험이 증가
    - 내화물 종류, 사이클, 차수에 따라 기준값과 작업 리스크가 달라짐
    """
    rng = np.random.default_rng(42)
    rows = 480
    temperature = rng.uniform(18, 42, rows)
    humidity = rng.uniform(35, 92, rows)
    amount_kg = rng.choice([350, 400, 450, 500], rows)
    cycle = rng.integers(1, 4, rows)
    round_no = rng.integers(1, 4, rows)
    material_code = rng.integers(0, 4, rows).astype(float)
    season_code = (temperature < 23).astype(float)
    target_water = (
        25.5
        + material_code * 0.18
        + (temperature < 23) * 0.25
        - np.maximum(humidity - 70, 0) * 0.025
        + rng.normal(0, 0.12, rows)
    )
    water = np.clip(target_water + rng.normal(0, 0.35, rows), 24.0, 28.0)
    mixing = np.clip(
        4.2
        + material_code * 0.25
        + (amount_kg - 350) / 300
        + rng.normal(0, 0.35, rows),
        3.5,
        7.0,
    )
    hopper_wait = np.clip(
        2.0
        + (round_no - 1) * 1.8
        + (cycle - 1) * 0.8
        + rng.normal(0, 2.2, rows),
        0,
        24,
    )
    retarder = (
        (
            (temperature >= 33)
            | ((humidity >= 78) & (hopper_wait >= 7))
            | ((round_no >= 3) & (material_code == 1))
        )
        & (rng.random(rows) > 0.18)
    ).astype(float)
    risk = (
        3.0
        + np.maximum(temperature - 27, 0) * 0.48
        + np.maximum(humidity - 65, 0) * 0.09
        + np.maximum(water - target_water - 0.35, 0) * 4.2
        + np.maximum(target_water - water - 0.55, 0) * 2.0
        + np.maximum(hopper_wait - 5, 0) * 0.42
        + np.maximum(mixing - 5.5, 0) * 1.2
        + (amount_kg - 350) * 0.012
        + (cycle - 1) * 0.7
        + (round_no - 1) * 0.9
        + material_code * 0.55
        - retarder * 2.8
        + rng.normal(0, 0.55, rows)
    )
    features = np.column_stack([
        temperature, humidity, amount_kg, cycle, round_no, material_code,
        water, mixing, hopper_wait, retarder, season_code
    ])
    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(features, risk)
    return model


DEMO_XGBOOST_MODEL = build_demo_xgboost_model()
ACTIVE_XGBOOST_MODEL = DEMO_XGBOOST_MODEL
TRAINING_SOURCE = "demo_seed"
TRAINING_ROWS = 240


def _training_features(row: dict) -> list[float]:
    return [
        float(row.get("temperature", 25)),
        float(row.get("humidity", 60)),
        float(row.get("amount_kg", row.get("amount", 500))),
        float(row.get("cycle", 1)),
        float(row.get("round", row.get("round_no", 1))),
        _material_code(str(row.get("material", ""))),
        float(row.get("water", row.get("actual_water_l", 26))),
        float(row.get("mixing", row.get("actual_mixing_min", 5))),
        float(row.get("hopper_wait", row.get("hopper_wait_min", 0))),
        float(row.get("retarder", row.get("retarder_used", 0))),
        float(row.get("season_code", 0)),
    ]


def _fit_field_xgboost(rows: list[dict]) -> XGBRegressor:
    if len(rows) < 8:
        raise HTTPException(status_code=422, detail="XGBoost 학습에는 최소 8건의 검증된 작업 데이터가 필요합니다.")
    X = np.asarray([_training_features(row) for row in rows], dtype=np.float32)
    y = np.asarray([float(row.get("risk", 0)) for row in rows], dtype=np.float32)
    model = XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y, verbose=False)
    return model


def _load_saved_xgboost_model() -> None:
    global ACTIVE_XGBOOST_MODEL, TRAINING_SOURCE, TRAINING_ROWS
    if not TRAINING_META_PATH.exists():
        return
    try:
        payload = json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        ACTIVE_XGBOOST_MODEL = _fit_field_xgboost(rows)
        TRAINING_SOURCE = "field_worklogs"
        TRAINING_ROWS = len(rows)
    except Exception:
        ACTIVE_XGBOOST_MODEL = DEMO_XGBOOST_MODEL
        TRAINING_SOURCE = "demo_seed"
        TRAINING_ROWS = 240


_load_saved_xgboost_model()


def recommendation_for_environment(
    temperature: float,
    humidity: float,
    season: str = "summer",
    amount_kg: float = 500,
    cycle: int = 1,
    round_no: int = 1,
    material: str = "저시멘트 캐스터블",
    hopper_wait: float = 0,
    retarder: float = 0,
) -> dict:
    season_code = 1.0 if season == "winter" else 0.0
    material_code = _material_code(material)
    base_water = 25.5 + material_code * 0.18 + (0.25 if season == "winter" else 0.0)
    water = round(max(24.0, min(28.0, base_water - max(0.0, humidity - 70.0) * 0.025 + max(0.0, temperature - 30.0) * 0.08)), 1)
    mixing_minutes = 5
    feature_row = np.array([[
        temperature, humidity, amount_kg, cycle, round_no, material_code,
        water, mixing_minutes, hopper_wait, retarder, season_code
    ]], dtype=float)
    predicted_risk = float(ACTIVE_XGBOOST_MODEL.predict(feature_row)[0])
    risk = round(max(3.0, min(30.0, predicted_risk)), 1)
    return {
        "risk": risk,
        "recommended_water_l": water,
        "mixing_minutes": 4 if risk >= 18 else 5,
        "retarder": risk > 10,
        "model": "XGBoost v2.4 · field recommendation",
        "season": season,
        "inputs": {
            "temperature": temperature,
            "humidity": humidity,
            "amount_kg": amount_kg,
            "cycle": cycle,
            "round": round_no,
            "material": material,
        },
    }


@app.get("/xgboost/status")
def xgboost_status() -> dict:
    return {
        "status": "ready",
        "model": "XGBoost",
        "model_version": "XGBoost v2.4",
        "training_source": TRAINING_SOURCE,
        "training_rows": TRAINING_ROWS,
        "message": "실제 작업일지 데이터가 등록되면 /xgboost/train으로 모델을 다시 학습합니다." if TRAINING_SOURCE == "demo_seed" else "검증된 작업일지 데이터로 학습된 모델입니다.",
    }


@app.post("/xgboost/train")
def xgboost_train(payload: dict = Body(...)) -> dict:
    global ACTIVE_XGBOOST_MODEL, TRAINING_SOURCE, TRAINING_ROWS
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="rows 배열이 필요합니다.")
    model = _fit_field_xgboost(rows)
    ACTIVE_XGBOOST_MODEL = model
    TRAINING_SOURCE = "field_worklogs"
    TRAINING_ROWS = len(rows)
    TRAINING_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_META_PATH.write_text(json.dumps({"rows": rows, "trained_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "trained",
        "model": "XGBoost",
        "model_version": "XGBoost v2.4",
        "training_source": TRAINING_SOURCE,
        "training_rows": TRAINING_ROWS,
        "message": "검증된 작업일지 데이터로 XGBoost 모델을 다시 학습했습니다.",
    }


@app.post("/xgboost/recommend")
def xgboost_recommend(payload: dict = Body(...)) -> dict:
    result = recommendation_for_environment(
        float(payload.get("temperature", 25)),
        float(payload.get("humidity", 60)),
        str(payload.get("season", "summer")),
        float(payload.get("amount_kg", 500)),
        int(payload.get("cycle", 1)),
        int(payload.get("round", payload.get("round_no", 1))),
        str(payload.get("material", "저시멘트 캐스터블")),
        float(payload.get("hopper_wait", 0)),
        float(payload.get("retarder", 0)),
    )
    return {**result, "model_version": "XGBoost v2.4", "training_source": TRAINING_SOURCE, "training_rows": TRAINING_ROWS}


@app.get("/sensor-state")
def sensor_state() -> dict:
    """PLC 전환 전 데모 센서 스트림. PLC 게이트웨이는 /sensor-ingest로 값을 보낸다."""
    if SENSOR_STATE["source"] == "demo-sensor":
        import math
        import time
        now = time.time()
        SENSOR_STATE["temperature"] = round(30.5 + math.sin(now / 18.0) * 1.8, 1)
        SENSOR_STATE["humidity"] = round(70.0 + math.sin(now / 25.0 + 0.8) * 8.0, 1)
        SENSOR_STATE["updated_at"] = datetime.now(timezone.utc).isoformat()
    temperature = float(SENSOR_STATE["temperature"])
    humidity = float(SENSOR_STATE["humidity"])
    season = str(SENSOR_STATE.get("season", "summer"))
    amount_kg = float(SENSOR_STATE.get("amount_kg", 500))
    cycle = int(SENSOR_STATE.get("cycle", 1))
    round_no = int(SENSOR_STATE.get("round", 1))
    material = str(SENSOR_STATE.get("material", "저시멘트 캐스터블"))
    return {
        "source": SENSOR_STATE["source"],
        "temperature": temperature,
        "humidity": humidity,
        "season": season,
        "updated_at": SENSOR_STATE["updated_at"],
        "amount_kg": amount_kg,
        "cycle": cycle,
        "round": round_no,
        "material": material,
        "recommendation": recommendation_for_environment(temperature, humidity, season, amount_kg, cycle, round_no, material),
    }


@app.post("/sensor-ingest")
def sensor_ingest(payload: dict = Body(...)) -> dict:
    """PLC/IoT 게이트웨이가 온도·습도를 전달하는 운영용 입력 API."""
    try:
        temperature = float(payload["temperature"])
        humidity = float(payload["humidity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="temperature와 humidity가 필요합니다.") from exc
    if not -20 <= temperature <= 100 or not 0 <= humidity <= 100:
        raise HTTPException(status_code=400, detail="센서 범위를 확인하세요.")
    SENSOR_STATE.update({
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "season": str(payload.get("season", "summer")) if str(payload.get("season", "summer")) in {"summer", "winter"} else "summer",
        "source": str(payload.get("source", "plc-gateway")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return sensor_state()


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

