"""
Weekly retraining script for the LSTM stock predictor.
Run: python -m scripts.train
Saves trained model to models/lstm_stock.pt
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.data_fetcher import LQ45_TICKERS, fetch_ohlcv
from app.services.feature_engineer import build_features, make_sequences, SEQUENCE_LEN, N_FEATURES
from app.models.lstm_predictor import StockLSTM

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_stock.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_training_data() -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    print(f"Fetching data for {len(LQ45_TICKERS)} tickers...")
    for i, ticker in enumerate(LQ45_TICKERS):
        data = fetch_ohlcv(ticker, period="5y")
        if data is None:
            print(f"  [{i+1}/{len(LQ45_TICKERS)}] {ticker}: no data, skipping")
            continue
        features = build_features(data["closes"], data["volumes"], data["timestamps"])
        # Target: normalised close of next day (index 0 of features is close_norm)
        targets = features[1:, 0]
        features = features[:-1]
        X, y = make_sequences(features, targets, SEQUENCE_LEN)
        all_X.append(X)
        all_y.append(y)
        print(f"  [{i+1}/{len(LQ45_TICKERS)}] {ticker}: {len(X)} sequences")

    if not all_X:
        raise RuntimeError("No training data collected")

    return np.concatenate(all_X), np.concatenate(all_y)


def train():
    print(f"Training on {DEVICE}")
    X, y = collect_training_data()
    print(f"Total sequences: {len(X)}")

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = StockLSTM(input_size=N_FEATURES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)
        scheduler.step()

        print(f"Epoch {epoch:2d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (val={val_loss:.4f})")

    print(f"\nTraining complete. Model saved to {MODEL_PATH}")
    _upload_to_hub(MODEL_PATH)


def _upload_to_hub(model_path: str) -> None:
    from app.config import MODEL_REPO, HF_TOKEN
    if not MODEL_REPO or not HF_TOKEN:
        print("HF_MODEL_REPO / HF_TOKEN not set — skipping HF Hub upload.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="lstm_stock.pt",
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message="Weekly LSTM retrain via GitHub Actions",
        )
        print(f"Model uploaded to HF Hub: {MODEL_REPO}/lstm_stock.pt")
    except Exception as e:
        print(f"HF Hub upload failed: {e}")


if __name__ == "__main__":
    train()
