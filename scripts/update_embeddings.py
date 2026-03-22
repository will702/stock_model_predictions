"""
Daily embedding update script.
Run: python -m scripts.update_embeddings
Saves embeddings to models/embeddings.json
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.data_fetcher import LQ45_TICKERS, fetch_ohlcv
from app.models.embeddings import compute_embedding

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# Rough sector mapping for LQ45 tickers (IDX sector codes 1-12)
SECTOR_MAP: dict[str, int] = {
    "BBCA": 4, "BBRI": 4, "BBNI": 4, "BMRI": 4, "BBTN": 4,  # banking
    "TLKM": 7, "EXCL": 7, "TBIG": 7, "TOWR": 7,              # telecom
    "ASII": 3, "ICBP": 2, "INDF": 2, "UNVR": 2, "SIDO": 2,  # consumer
    "ANTM": 5, "TINS": 5, "PTBA": 5, "ADRO": 5, "ITMG": 5,  # mining
    "INCO": 5, "MDKA": 5, "HRUM": 5, "MEDC": 5,
    "SMGR": 1, "INTP": 1,                                      # cement/basic materials
    "KLBF": 6, "MIKA": 6,                                      # healthcare
    "JSMR": 8, "WIKA": 8, "WSKT": 8,                          # infra
    "AALI": 9, "CPIN": 9, "JPFA": 9,                          # agri
    "MAPI": 2, "AMRT": 2, "ERAA": 2,                          # retail
    "PGAS": 10, "PGEO": 10,                                    # energy
    "BUKA": 11, "GOTO": 11, "EMTK": 11,                       # tech
    "BRPT": 1, "INKP": 1,                                      # chemicals/paper
    "SRTG": 12, "MNCN": 12,                                    # conglomerate/media
    "MBMA": 5, "SMRA": 8,
}


def update():
    embeddings: dict[str, list[float]] = {}

    # Load existing to preserve tickers not in current batch
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH) as f:
            embeddings = json.load(f)

    print(f"Updating embeddings for {len(LQ45_TICKERS)} tickers...")
    updated = 0
    for i, ticker in enumerate(LQ45_TICKERS):
        data = fetch_ohlcv(ticker, period="2y")
        if data is None:
            print(f"  [{i+1}/{len(LQ45_TICKERS)}] {ticker}: no data, skipping")
            continue
        sector_id = SECTOR_MAP.get(ticker, 0)
        emb = compute_embedding(data["closes"], data["volumes"], sector_id)
        key = f"{ticker}.JK"
        embeddings[key] = emb.tolist()
        updated += 1
        print(f"  [{i+1}/{len(LQ45_TICKERS)}] {ticker}: ok")

    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(embeddings, f)

    print(f"\nDone. {updated} embeddings saved to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    update()
