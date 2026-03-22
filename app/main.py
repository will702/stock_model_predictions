"""StockPro ML Backend — FastAPI app for Hugging Face Spaces."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, similarity, anomaly

app = FastAPI(
    title="StockPro ML Backend",
    description="LSTM price prediction, stock similarity, and Isolation Forest anomaly detection for IDX stocks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["prediction"])
app.include_router(similarity.router, tags=["similarity"])
app.include_router(anomaly.router, tags=["anomaly"])


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
