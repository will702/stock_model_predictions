"""Feature engineering for LSTM price prediction."""
import numpy as np
import pandas as pd
from typing import Tuple


SEQUENCE_LEN = 30  # lookback window
FEATURE_COLS = ["close_norm", "volume_norm", "rsi", "macd_norm", "bb_width", "day_sin", "day_cos"]
N_FEATURES = len(FEATURE_COLS)


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-9))
    return np.clip(rs / (1 + rs), 0, 1)  # normalised 0-1


def _macd(prices: np.ndarray) -> np.ndarray:
    s = pd.Series(prices)
    macd_line = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal).values


def _bollinger_width(prices: np.ndarray, period: int = 20) -> np.ndarray:
    s = pd.Series(prices)
    sma = s.rolling(period, min_periods=1).mean()
    std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0)
    mid = sma.where(sma != 0, 1)
    return (2 * std / mid).fillna(0).values


def build_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,  # unix seconds
) -> np.ndarray:
    """Return (T, N_FEATURES) feature matrix, normalised."""
    n = len(closes)

    # Price normalisation: rolling 30-day z-score
    s_close = pd.Series(closes)
    roll_mean = s_close.rolling(30, min_periods=1).mean().values
    roll_std = s_close.rolling(30, min_periods=1).std(ddof=0).fillna(1).values
    roll_std = np.where(roll_std == 0, 1, roll_std)
    close_norm = (closes - roll_mean) / roll_std

    # Volume normalisation
    s_vol = pd.Series(volumes.astype(float))
    v_mean = s_vol.rolling(30, min_periods=1).mean().values
    v_std = s_vol.rolling(30, min_periods=1).std(ddof=0).fillna(1).values
    v_std = np.where(v_std == 0, 1, v_std)
    volume_norm = (volumes - v_mean) / v_std

    rsi = _rsi(closes)
    macd_raw = _macd(closes)
    macd_std = np.std(macd_raw) or 1
    macd_norm = macd_raw / macd_std
    bb_width = _bollinger_width(closes)

    # Cyclical day-of-week encoding
    days = (pd.to_datetime(timestamps, unit="s").dayofweek.values).astype(float)
    day_sin = np.sin(2 * np.pi * days / 5)
    day_cos = np.cos(2 * np.pi * days / 5)

    features = np.stack([close_norm, volume_norm, rsi, macd_norm, bb_width, day_sin, day_cos], axis=1)
    return features.astype(np.float32)


def make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int = SEQUENCE_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows over features to produce (X, y) training pairs."""
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
