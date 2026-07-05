"""
main.py — FastAPI backend za detekciju pada
============================================
Pokretanje:
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /predict
    Body:  { "model": "gnb"|"knn"|"mlp", "features": [17 floatova] }
    Resp:  { "label": 0|1, "label_text": "ADL"|"FALL",
             "confidence": float, "model_used": str, "inference_ms": float }

    GET  /health
    Resp:  { "status": "ok", "models_loaded": [...] }
"""

import time
import json
import logging
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONSTANTE
# ─────────────────────────────────────────────

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
N_FEATURES     = 17
MLP_THRESHOLD  = 0.3   # isti kao u treningu

# ─────────────────────────────────────────────
# UČITAVANJE MODELA (jednom, pri startu)
# ─────────────────────────────────────────────

log.info("Učitavanje modela i skalera...")

try:
    _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    _gnb    = joblib.load(os.path.join(MODELS_DIR, "gnb.pkl"))
    _knn    = joblib.load(os.path.join(MODELS_DIR, "knn.pkl"))
    _mlp    = tf.keras.models.load_model(os.path.join(MODELS_DIR, "mlp.keras"))

    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        FEATURE_NAMES = json.load(f)

    log.info(f"✓ Svi modeli učitani  |  Features: {N_FEATURES}")
except Exception as e:
    log.error(f"GREŠKA pri učitavanju modela: {e}")
    raise

LOADED_MODELS = ["gnb", "knn", "mlp"]

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Fall Detection API",
    description="Detekcija pada zasnovana na SisFall ML modelima.",
    version="1.0.0",
)

# Dozvoli sve originale (za razvoj; u produkciji ograničiti)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ŠEME
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    model:    Literal["gnb", "knn", "mlp"]
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(
                f"Očekivano {N_FEATURES} feature-a, primljeno {len(v)}. "
                f"Redosled: {FEATURE_NAMES}"
            )
        return v


class PredictResponse(BaseModel):
    label:        int          # 0 = ADL, 1 = FALL
    label_text:   str          # "ADL" ili "FALL"
    confidence:   float        # verovatnoća klase 1 (Fall)
    model_used:   str
    inference_ms: float


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": LOADED_MODELS,
        "n_features":    N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "mlp_threshold": MLP_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Standardizacija (isti scaler iz treninga)
    x_raw = np.array(req.features, dtype=np.float64).reshape(1, -1)
    try:
        x_scaled = _scaler.transform(x_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler greška: {e}")

    # 2. Predikcija
    t0 = time.perf_counter()
    try:
        if req.model == "gnb":
            prob  = float(_gnb.predict_proba(x_scaled)[0][1])
            label = int(prob >= 0.5)
        elif req.model == "knn":
            prob  = float(_knn.predict_proba(x_scaled)[0][1])
            label = int(prob >= 0.5)
        else:  # mlp
            prob  = float(_mlp.predict(x_scaled, verbose=0)[0][0])
            label = int(prob >= MLP_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Greška predikcije ({req.model}): {e}")

    inference_ms = (time.perf_counter() - t0) * 1000

    log.info(
        f"model={req.model}  label={label}  conf={prob:.4f}  {inference_ms:.2f}ms"
    )

    return PredictResponse(
        label        = label,
        label_text   = "FALL" if label == 1 else "ADL",
        confidence   = round(prob, 6),
        model_used   = req.model,
        inference_ms = round(inference_ms, 3),
    )
