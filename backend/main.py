from pathlib import Path
import json
from typing import Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "ml" / "models"
FRONTEND_DIR = ROOT / "frontend"

# Load artifacts once when the API starts. Keeping the trained model in memory
# makes prediction fast and avoids reloading files on every request.
MODEL = joblib.load(MODEL_DIR / "intrusion_model.joblib")
PREPROCESSOR = joblib.load(MODEL_DIR / "preprocessor.joblib")
with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as f:
    FEATURE_INFO = json.load(f)
with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURES = FEATURE_INFO["raw_features"]

class TrafficRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: float = Field(ge=0)
    protocol_type: str
    service: str
    flag: str
    src_bytes: float = Field(ge=0)
    dst_bytes: float = Field(ge=0)
    land: float = Field(ge=0)
    wrong_fragment: float = Field(ge=0)
    urgent: float = Field(ge=0)
    hot: float = Field(ge=0)
    num_failed_logins: float = Field(ge=0)
    logged_in: float = Field(ge=0)
    num_compromised: float = Field(ge=0)
    root_shell: float = Field(ge=0)
    su_attempted: float = Field(ge=0)
    num_root: float = Field(ge=0)
    num_file_creations: float = Field(ge=0)
    num_shells: float = Field(ge=0)
    num_access_files: float = Field(ge=0)
    num_outbound_cmds: float = Field(ge=0)
    is_host_login: float = Field(ge=0)
    is_guest_login: float = Field(ge=0)
    count: float = Field(ge=0)
    srv_count: float = Field(ge=0)
    serror_rate: float = Field(ge=0, le=1)
    srv_serror_rate: float = Field(ge=0, le=1)
    rerror_rate: float = Field(ge=0, le=1)
    srv_rerror_rate: float = Field(ge=0, le=1)
    same_srv_rate: float = Field(ge=0, le=1)
    diff_srv_rate: float = Field(ge=0, le=1)
    srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_count: float = Field(ge=0)
    dst_host_srv_count: float = Field(ge=0)
    dst_host_same_srv_rate: float = Field(ge=0, le=1)
    dst_host_diff_srv_rate: float = Field(ge=0, le=1)
    dst_host_same_src_port_rate: float = Field(ge=0, le=1)
    dst_host_srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_serror_rate: float = Field(ge=0, le=1)
    dst_host_srv_serror_rate: float = Field(ge=0, le=1)
    dst_host_rerror_rate: float = Field(ge=0, le=1)
    dst_host_srv_rerror_rate: float = Field(ge=0, le=1)

app = FastAPI(title="CypherX NIDS API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static assets are served by FastAPI so the user only needs ONE server.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "CypherX NIDS API",
        "model": METADATA["model"],
        "features": len(FEATURES),
    }

@app.get("/model-info")
def model_info():
    return METADATA

@app.post("/predict-batch")
def predict_batch(records: List[TrafficRecord]):
    if not records:
        raise HTTPException(status_code=422, detail="At least one traffic record is required.")
    if len(records) > 10:
        raise HTTPException(status_code=422, detail="A maximum of 10 traffic records can be classified at once.")

    try:
        rows = [record.model_dump() for record in records]
        df = pd.DataFrame([[row[name] for name in FEATURES] for row in rows], columns=FEATURES)
        X = PREPROCESSOR.transform(df)
        predictions = MODEL.predict(X)
        probabilities = MODEL.predict_proba(X)
        results = []
        for index, (pred, probs) in enumerate(zip(predictions, probabilities), start=1):
            pred = int(pred)
            results.append({
                "index": index,
                "prediction": "Attack" if pred == 1 else "Normal",
                "attack_category": "Attack" if pred == 1 else "Normal",
                "confidence": round(float(probs[pred]), 6),
                "model": METADATA["model"],
                "features_received": len(rows[index - 1]),
            })
        return {"count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction pipeline error: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/predict")
def predict(record: TrafficRecord):
    try:
        row = record.model_dump()
        df = pd.DataFrame([[row[name] for name in FEATURES]], columns=FEATURES)
        X = PREPROCESSOR.transform(df)
        pred = int(MODEL.predict(X)[0])
        probabilities = MODEL.predict_proba(X)[0]
        confidence = float(probabilities[pred])
        prediction = "Attack" if pred == 1 else "Normal"

        return {
            "prediction": prediction,
            "attack_category": "Attack" if pred == 1 else "Normal",
            "confidence": round(confidence, 6),
            "model": METADATA["model"],
            "features_received": len(row),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction pipeline error: {type(exc).__name__}: {exc}",
        ) from exc
