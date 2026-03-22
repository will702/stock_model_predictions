"""LSTM price predictor — small CPU-friendly model."""
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from app.services.feature_engineer import N_FEATURES, SEQUENCE_LEN, build_features


class StockLSTM(nn.Module):
    def __init__(self, input_size: int = N_FEATURES, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


_model: Optional[StockLSTM] = None
_model_path: Optional[str] = None


def _maybe_download_model(model_path: str) -> bool:
    """Download model from HF Hub if not present locally. Returns True if available."""
    if os.path.exists(model_path):
        return True
    import app.config as cfg
    if not cfg.MODEL_REPO or not cfg.HF_TOKEN:
        return False
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=cfg.MODEL_REPO,
            filename="lstm_stock.pt",
            token=cfg.HF_TOKEN,
            local_dir=os.path.dirname(model_path),
        )
        # hf_hub_download may save to a cache path; copy to expected location
        if local != model_path:
            import shutil
            shutil.copy2(local, model_path)
        return os.path.exists(model_path)
    except Exception as e:
        print(f"[lstm] Could not download model from HF Hub: {e}")
        return False


def load_model(model_path: str) -> StockLSTM:
    global _model, _model_path
    if _model is not None and _model_path == model_path:
        return _model
    _maybe_download_model(model_path)
    model = StockLSTM()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _model = model
    _model_path = model_path
    return model


def predict_returns(
    closes: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    days: int,
    model_path: str,
) -> dict:
    """
    Auto-regressively predict `days` forward log-returns.
    Returns predictions as price levels.
    """
    model = load_model(model_path)
    features = build_features(closes, volumes, timestamps)

    if len(features) < SEQUENCE_LEN:
        raise ValueError(f"Need at least {SEQUENCE_LEN} candles, got {len(features)}")

    window = features[-SEQUENCE_LEN:].copy()  # (SEQ_LEN, N_FEATURES)
    current_price = float(closes[-1])

    # Rolling std/mean of last 30 closes for denormalisation
    roll_mean = float(np.mean(closes[-30:]))
    roll_std = float(np.std(closes[-30:])) or 1.0

    predictions = []
    lower_bound = []
    upper_bound = []

    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(window[None], dtype=torch.float32)  # (1, SEQ_LEN, F)
            z_pred = model(x).item()  # predicted normalised close

            # Denormalise: z_pred = (price - mean) / std  →  price = z*std + mean
            price_pred = z_pred * roll_std + roll_mean

            # Confidence interval widens with horizon (±1.5% per day baseline)
            horizon = len(predictions) + 1
            margin = price_pred * 0.015 * np.sqrt(horizon)
            predictions.append(round(price_pred, 2))
            lower_bound.append(round(price_pred - margin, 2))
            upper_bound.append(round(price_pred + margin, 2))

            # Shift window: drop oldest, append new feature row
            new_z = (price_pred - roll_mean) / roll_std
            new_row = window[-1].copy()
            new_row[0] = new_z  # update close_norm; keep other features constant
            window = np.vstack([window[1:], new_row[None]])

    final_price = predictions[-1]
    trend = "bullish" if final_price > current_price * 1.005 else \
            "bearish" if final_price < current_price * 0.995 else "neutral"
    change_pct = (final_price - current_price) / current_price * 100

    return {
        "method": "lstm",
        "predictions": predictions,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "target_price": final_price,
        "trend": trend,
        "change_pct": round(change_pct, 2),
        "confidence": 72,  # fixed; could be calibrated from val loss
        "support": round(min(lower_bound), 2),
        "resistance": round(max(upper_bound), 2),
    }
