"""3A — LSTM price prediction endpoint."""
import os
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, field_validator

from app.config import API_KEY, MODEL_DIR
from app.services.data_fetcher import fetch_ohlcv
from app.models.lstm_predictor import predict_returns

router = APIRouter()
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_stock.pt")


class PredictRequest(BaseModel):
    symbol: str
    days: int = 7
    # Optional pre-fetched OHLCV arrays (avoids cloud-IP blocking of Yahoo Finance)
    closes: Optional[List[float]] = None
    volumes: Optional[List[float]] = None
    timestamps: Optional[List[int]] = None

    @field_validator("days")
    @classmethod
    def clamp_days(cls, v: int) -> int:
        return max(1, min(v, 30))

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.strip().upper()


@router.post("/predict")
async def predict(req: PredictRequest, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet. Run scripts/train.py first.")

    # Use caller-supplied data if provided, otherwise fetch from Yahoo Finance
    if req.closes and req.volumes and req.timestamps and len(req.closes) >= 60:
        data = {
            "closes": np.array(req.closes, dtype=np.float32),
            "volumes": np.array(req.volumes, dtype=np.float64),
            "timestamps": np.array(req.timestamps, dtype=np.int64),
            "symbol": req.symbol,
        }
    else:
        data = fetch_ohlcv(req.symbol)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for {req.symbol}")

    try:
        result = predict_returns(
            closes=data["closes"],
            volumes=data["volumes"],
            timestamps=data["timestamps"],
            days=req.days,
            model_path=MODEL_PATH,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"symbol": req.symbol, **result}
