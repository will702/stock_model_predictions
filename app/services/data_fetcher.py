"""Fetch OHLCV data from Yahoo Finance for IDX stocks."""
import requests
import yfinance as yf
import numpy as np
from typing import Optional


IDX_SUFFIX = ".JK"
LQ45_TICKERS = [
    "AALI", "ADRO", "AKRA", "AMRT", "ANTM", "ASII", "BBCA", "BBNI", "BBRI", "BBTN",
    "BMRI", "BRPT", "BUKA", "CPIN", "EMTK", "ERAA", "EXCL", "GOTO", "HRUM", "ICBP",
    "INCO", "INDF", "INKP", "INTP", "ITMG", "JPFA", "JSMR", "KLBF", "MAPI", "MBMA",
    "MDKA", "MEDC", "MIKA", "MNCN", "PGAS", "PGEO", "PTBA", "SIDO", "SMGR", "SMRA",
    "SRTG", "TBIG", "TINS", "TLKM", "TOWR", "UNTR", "UNVR", "WIKA", "WSKT",
]

# Browser-like session to avoid Yahoo Finance rate-limiting cloud IPs
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
})


def normalise_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if not symbol.endswith(IDX_SUFFIX):
        symbol += IDX_SUFFIX
    return symbol


def fetch_ohlcv(symbol: str, period: str = "2y") -> Optional[dict]:
    """Return dict with closes, volumes, timestamps arrays. None on failure."""
    ticker = normalise_ticker(symbol)
    try:
        df = yf.Ticker(ticker, session=_session).history(period=period)
        if df.empty or len(df) < 60:
            return None
        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < 60:
            return None
        return {
            "closes": df["Close"].values.astype(np.float32),
            "volumes": df["Volume"].values.astype(np.float64),
            "timestamps": df.index.tz_convert(None).astype(np.int64) // 10**9,
            "symbol": ticker,
        }
    except Exception as e:
        print(f"  fetch_ohlcv({ticker}): {e}")
        return None


def fetch_batch(tickers: list, period: str = "2y") -> dict:
    """Fetch multiple tickers; skip failures silently."""
    results = {}
    for t in tickers:
        data = fetch_ohlcv(t, period)
        if data:
            results[t] = data
    return results
