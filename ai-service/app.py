"""AI 스마트 믹싱용 현장 이미지 분석 API.

현재는 정상 이미지가 충분히 쌓이기 전까지 기준 이미지 기반 시범 판정을 제공합니다.
정상 이미지가 30장 이상 쌓이면 /train-patchcore를 호출해 PatchCore 모델을 학습하도록
확장할 수 있습니다. 서버 관리자 키나 Supabase 비밀 키는 이 서비스에 저장하지 않습니다.
"""

from __future__ import annotations

import io
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
    "updated_at": datetime.now(timezone.utc).isoformat(),
}


def build_demo_xgboost_model() -> XGBRegressor:
    """발표용 더미 작업 데이터로 XGBoost 위험도 예측 모델을 준비한다."""
    rng = np.random.default_rng(42)
    rows = 240
    temperature = rng.uniform(18, 42, rows)
    humidity = rng.uniform(35, 92, rows)
    water = rng.uniform(24, 28, rows)
    mixing = rng.integers(4, 8, rows)
    hopper_wait = rng.uniform(0, 24, rows)
    retarder = rng.integers(0, 2, rows)
    season_code = rng.integers(0, 2, rows)
    risk = (
        4.5
        + np.maximum(temperature - 26, 0) * 0.55
        + np.maximum(humidity - 58, 0) * 0.11
        + np.maximum(water - 26.2, 0) * 1.8
        + np.maximum(hopper_wait - 5, 0) * 0.22
        + retarder * -2.0
        + season_code * 0.8
        + rng.normal(0, 0.8, rows)
    )
    features = np.column_stack([temperature, humidity, water, mixing, hopper_wait, retarder, season_code])
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(features, risk)
    return model


DEMO_XGBOOST_MODEL = build_demo_xgboost_model()


def recommendation_for_environment(temperature: float, humidity: float, season: str = "summer") -> dict:
    season_code = 1.0 if season == "winter" else 0.0
    seasonal_offset = 0.3 if season == "winter" else -0.2
    water = round(max(24.0, min(28.0, 26.0 + seasonal_offset + max(0.0, temperature - 30.0) * 0.18 - max(0.0, humidity - 70.0) * 0.04)), 1)
    mixing_minutes = 5
    feature_row = np.array([[temperature, humidity, water, mixing_minutes, 0.0, 0.0, season_code]], dtype=float)
    predicted_risk = float(DEMO_XGBOOST_MODEL.predict(feature_row)[0])
    risk = round(max(5.0, min(30.0, predicted_risk)), 1)
    return {
        "risk": risk,
        "recommended_water_l": water,
        "mixing_minutes": 4 if risk >= 18 else 5,
        "retarder": risk > 10,
        "model": "XGBoost dummy model v2 · seasonal",
        "season": season,
    }


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
    return {
        "source": SENSOR_STATE["source"],
        "temperature": temperature,
        "humidity": humidity,
        "season": season,
        "updated_at": SENSOR_STATE["updated_at"],
        "recommendation": recommendation_for_environment(temperature, humidity, season),
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

