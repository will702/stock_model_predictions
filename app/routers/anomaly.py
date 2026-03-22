"""3C — Isolation Forest multi-dimensional anomaly detection."""
import os
import numpy as np
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, field_validator

from app.config import API_KEY
from app.services.data_fetcher import fetch_ohlcv
from app.services.feature_engineer import build_features

router = APIRouter()


class IsolationTree:
    """Minimal Isolation Tree — pure Python/NumPy, no sklearn dependency."""

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.split_feature: int | None = None
        self.split_value: float | None = None
        self.left: "IsolationTree | None" = None
        self.right: "IsolationTree | None" = None
        self.size: int = 0

    def fit(self, X: np.ndarray, depth: int = 0) -> None:
        self.size = len(X)
        if depth >= self.max_depth or len(X) <= 1:
            return
        n_features = X.shape[1]
        self.split_feature = int(np.random.randint(0, n_features))
        col = X[:, self.split_feature]
        lo, hi = col.min(), col.max()
        if lo == hi:
            return
        self.split_value = float(np.random.uniform(lo, hi))
        mask = col < self.split_value
        self.left = IsolationTree(self.max_depth)
        self.right = IsolationTree(self.max_depth)
        self.left.fit(X[mask], depth + 1)
        self.right.fit(X[~mask], depth + 1)

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        if self.split_feature is None or self.split_value is None:
            return depth + _c(self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1) if self.left else depth + 1.0
        return self.right.path_length(x, depth + 1) if self.right else depth + 1.0


def _c(n: int) -> float:
    """Expected path length of unsuccessful BST search."""
    if n <= 1:
        return 0.0
    h = np.log(n - 1) + 0.5772156649  # Euler-Mascheroni
    return 2 * h - 2 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_trees: int = 100, sample_size: int = 256, max_depth: int = 8):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.trees: list[IsolationTree] = []
        self._c: float = 1.0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        n = len(X)
        sz = min(self.sample_size, n)
        self._c = _c(sz)
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, sz, replace=False)
            tree = IsolationTree(self.max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score in [0, 1]; higher = more anomalous."""
        scores = []
        for x in X:
            avg_path = np.mean([t.path_length(x) for t in self.trees])
            score = 2 ** (-avg_path / (self._c or 1.0))
            scores.append(score)
        return np.array(scores, dtype=np.float32)


class AnomalyRequest(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.strip().upper()


@router.post("/anomaly-ml")
async def anomaly_ml(req: AnomalyRequest, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = fetch_ohlcv(req.symbol, period="6mo")
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data found for {req.symbol}")

    closes = data["closes"]
    volumes = data["volumes"]
    timestamps = data["timestamps"]

    if len(closes) < 40:
        raise HTTPException(status_code=422, detail="Need at least 40 candles for ML anomaly detection")

    features = build_features(closes, volumes, timestamps)

    forest = IsolationForest(n_trees=100, sample_size=min(256, len(features)))
    forest.fit(features)
    scores = forest.anomaly_scores(features)

    # Flag top anomalies (score > 0.65)
    threshold = 0.65
    anomalies = []
    for i, score in enumerate(scores):
        if score > threshold:
            # Map back to date from timestamps
            ts = int(timestamps[i]) if i < len(timestamps) else 0
            from datetime import datetime, timezone
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            anomalies.append({
                "date": date_str,
                "score": round(float(score), 4),
                "severity": "high" if score > 0.75 else "medium",
                "description": f"Multi-dimensional anomaly detected (IF score {score:.2f})",
            })

    # Sort by score descending, return top 10
    anomalies.sort(key=lambda x: x["score"], reverse=True)
    anomalies = anomalies[:10]

    return {
        "symbol": req.symbol,
        "ml_anomalies": anomalies,
        "ml_score": round(float(np.mean(scores[-20:])), 4),  # recent 20 bars avg score
        "total": len(anomalies),
    }
