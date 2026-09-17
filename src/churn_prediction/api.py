from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import FEATURE_COLUMNS
from .settings import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
app = FastAPI(title="E-commerce Churn Prediction API", version="1.0.0")


class UserFeatures(BaseModel):
    user_id: str = "anonymous"
    recency_days: float = Field(0, ge=0); frequency_90d: float = Field(0, ge=0); monetary_90d: float = Field(0, ge=0)
    events_30d: float = Field(0, ge=0); events_90d: float = Field(0, ge=0); clicks_30d: float = Field(0, ge=0)
    browses_30d: float = Field(0, ge=0); carts_30d: float = Field(0, ge=0); purchases_30d: float = Field(0, ge=0)
    active_days_30d: float = Field(0, ge=0); avg_order_amount_90d: float = Field(0, ge=0); days_since_register: float = Field(0, ge=0)
    city: str = "unknown"; channel: str = "unknown"


@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: UserFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="模型不存在，请先运行 python scripts/run_pipeline.py")
    pipeline = joblib.load(MODEL_PATH); data = payload.model_dump(); user_id = data.pop("user_id")
    frame = pd.DataFrame([{column: data[column] for column in FEATURE_COLUMNS}])
    probability = float(pipeline.predict_proba(frame)[0, 1])
    return {"user_id": user_id, "churn_probability": round(probability, 6), "is_high_risk": probability >= 0.5, "threshold": 0.5}
