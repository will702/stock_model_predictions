"""3B — Stock similarity endpoint."""
import os
import json
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, field_validator

from app.config import API_KEY, MODEL_DIR
from app.services.data_fetcher import fetch_ohlcv
from app.models.embeddings import compute_embedding, find_similar

router = APIRouter()
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.json")


def _load_embeddings() -> dict[str, list[float]]:
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}
    with open(EMBEDDINGS_PATH) as f:
        return json.load(f)


class SimilarRequest(BaseModel):
    symbol: str
    top_n: int = 6
    sector_id: int = 0

    @field_validator("top_n")
    @classmethod
    def clamp_top_n(cls, v: int) -> int:
        return max(1, min(v, 20))

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.strip().upper()


@router.post("/similar")
async def similar(req: SimilarRequest, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    all_embeddings = _load_embeddings()
    if not all_embeddings:
        raise HTTPException(status_code=503, detail="Embeddings not built yet. Run scripts/update_embeddings.py first.")

    # Use pre-computed embedding if available, else compute on-the-fly
    symbol_key = req.symbol.upper()
    if symbol_key in all_embeddings or f"{symbol_key}.JK" in all_embeddings:
        key = symbol_key if symbol_key in all_embeddings else f"{symbol_key}.JK"
        target_emb = all_embeddings[key]
    else:
        data = fetch_ohlcv(req.symbol)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for {req.symbol}")
        emb = compute_embedding(data["closes"], data["volumes"], req.sector_id)
        target_emb = emb.tolist()

    results = find_similar(target_emb, all_embeddings, top_n=req.top_n, exclude_symbol=req.symbol)
    return {"symbol": req.symbol, "similar": results, "total": len(results)}
