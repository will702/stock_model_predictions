"""
Stock similarity via 16-dimensional feature embeddings.
Stored in Supabase as JSONB float arrays (no pgvector required).
Cosine similarity computed here for the /similar endpoint.
"""
import numpy as np
from typing import Optional
import pandas as pd


EMBED_DIM = 16


def _safe_norm(v: np.ndarray) -> float:
    return float(np.std(v)) or 1.0


def compute_embedding(
    closes: np.ndarray,
    volumes: np.ndarray,
    sector_id: int = 0,
) -> np.ndarray:
    """
    16-dim embedding per stock:
    [0]   sector (normalised 0-1)
    [1]   annualised return (capped ±100%)
    [2]   annualised volatility
    [3-6] return autocorrelations lag 1-4
    [7]   beta proxy (corr with overall index, approx)
    [8]   volume z-score trend (avg recent vs avg older)
    [9]   max drawdown
    [10]  skewness of returns
    [11]  kurtosis of returns
    [12-15] rolling return quartiles (q25, q50, q75, q90)
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)

    if len(closes) < 30:
        return vec

    ret = np.diff(np.log(closes + 1e-9))
    T = len(ret)

    # [0] sector normalised 0-1 (max ~12 sectors on IDX)
    vec[0] = min(sector_id / 12.0, 1.0)

    # [1] annualised return, capped
    ann_ret = np.mean(ret) * 252
    vec[1] = float(np.clip(ann_ret, -1.0, 1.0))

    # [2] annualised volatility
    vec[2] = float(np.std(ret) * np.sqrt(252))

    # [3-6] autocorrelations lag 1-4
    for lag in range(1, 5):
        if T > lag + 1:
            corr = float(np.corrcoef(ret[:-lag], ret[lag:])[0, 1])
            vec[2 + lag] = corr if not np.isnan(corr) else 0.0

    # [7] beta proxy: correlation with its own 20-day rolling avg (smoother = lower beta)
    s = pd.Series(closes)
    trend = s.rolling(20, min_periods=5).mean().dropna().values
    if len(trend) > 10:
        corr = float(np.corrcoef(closes[-len(trend):], trend)[0, 1])
        vec[7] = corr if not np.isnan(corr) else 0.0

    # [8] volume trend: mean of recent 20 vs older 20
    if len(volumes) >= 40:
        recent = float(np.mean(volumes[-20:]))
        older = float(np.mean(volumes[-40:-20])) or 1.0
        vec[8] = float(np.clip((recent / older) - 1, -1, 1))

    # [9] max drawdown
    cum = np.cumprod(1 + ret)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / (running_max + 1e-9)
    vec[9] = float(np.min(drawdowns))

    # [10] skewness
    mu, sigma = np.mean(ret), np.std(ret) or 1
    vec[10] = float(np.clip(np.mean(((ret - mu) / sigma) ** 3), -3, 3))

    # [11] excess kurtosis
    vec[11] = float(np.clip(np.mean(((ret - mu) / sigma) ** 4) - 3, -3, 3))

    # [12-15] return quartiles normalised by vol
    if sigma > 0:
        qs = np.quantile(ret, [0.25, 0.5, 0.75, 0.90])
        vec[12:16] = np.clip(qs / sigma, -3, 3).astype(np.float32)

    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def find_similar(
    target_embedding: list[float],
    all_embeddings: dict[str, list[float]],
    top_n: int = 6,
    exclude_symbol: Optional[str] = None,
) -> list[dict]:
    """Return top_n most similar stocks by cosine similarity."""
    target = np.array(target_embedding, dtype=np.float32)
    scores = []
    for symbol, emb in all_embeddings.items():
        if exclude_symbol and symbol.upper() == exclude_symbol.upper():
            continue
        sim = cosine_similarity(target, np.array(emb, dtype=np.float32))
        scores.append({"symbol": symbol, "similarity": round(sim, 4)})
    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return scores[:top_n]
