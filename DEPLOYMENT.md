# ML Backend Deployment Guide

## Overview

```
Local machine  →  train once  →  HF Hub (model storage)
GitHub Actions →  retrain weekly  →  HF Hub
GitHub Actions →  embeddings daily  →  repo commit
HF Spaces      →  FastAPI app  →  downloads model from HF Hub on boot
Next.js (Vercel) →  proxies requests  →  HF Space URL
```

---

## Prerequisites

- Hugging Face account → https://huggingface.co/join
- Python 3.11+ installed locally
- Git + GitHub repository for this project
- Vercel project already deployed (for the Next.js app)

---

## Step 1 — Create Hugging Face Tokens and Repos

### 1.1 Create a write token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: `stockpro-actions`
4. Role: **Write**
5. Click **Generate token**
6. **Copy and save** the token — you'll need it several times below

### 1.2 Create the model repository (for storing the .pt file)

1. Go to https://huggingface.co/new-model
2. Model name: `stockpro-lstm`
3. Visibility: **Private**
4. Click **Create model**
5. Note the full repo ID: `your-username/stockpro-lstm`

### 1.3 Create the Space (for the FastAPI app)

1. Go to https://huggingface.co/new-space
2. Space name: `stockpro-ml`
3. SDK: **Docker**
4. Visibility: **Private** (or Public if you prefer)
5. Click **Create Space**
6. Note the Space URL: `https://your-username-stockpro-ml.hf.space`

---

## Step 2 — Set Up Local Python Environment

```bash
cd ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch CPU-only (faster install, sufficient for training small models)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
pip install fastapi uvicorn pydantic numpy pandas yfinance \
            scikit-learn joblib huggingface_hub python-dotenv
```

### 2.1 Create local `.env` file

```bash
# ml/.env  (never commit this file)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
HF_MODEL_REPO=your-username/stockpro-lstm
ML_BACKEND_API_KEY=choose-a-random-secret-string
```

---

## Step 3 — Train the Model Locally (First Time)

This only needs to be done once. GitHub Actions handles future retraining.

```bash
cd ml
python -m scripts.train
```

Expected output:
```
Fetching data for 49 tickers...
  [1/49] AALI: 1180 sequences
  [2/49] ADRO: 1204 sequences
  ...
Epoch  1/30  train=0.0821  val=0.0934
Epoch  2/30  train=0.0743  val=0.0871
  ✓ Saved best model (val=0.0871)
...
Epoch 30/30  train=0.0412  val=0.0523
Training complete. Model saved to models/lstm_stock.pt
Model uploaded to HF Hub: your-username/stockpro-lstm/lstm_stock.pt
```

Training takes ~15-30 minutes on CPU. The model is automatically uploaded to HF Hub at the end.

---

## Step 4 — Generate Embeddings Locally (First Time)

```bash
cd ml
python -m scripts.update_embeddings
```

Expected output:
```
Updating embeddings for 49 tickers...
  [1/49] AALI: ok
  ...
Done. 49 embeddings saved to models/embeddings.json
```

Then commit the embeddings file to the repo:

```bash
git add models/embeddings.json
git commit -m "chore: initial stock embeddings"
git push
```

---

## Step 5 — Deploy the FastAPI App to HF Spaces

### 5.1 Add Space secrets

In your HF Space settings (https://huggingface.co/spaces/your-username/stockpro-ml/settings):

1. Click **New secret** for each:

| Secret name | Value |
|-------------|-------|
| `HF_TOKEN` | Your HF write token from Step 1.1 |
| `HF_MODEL_REPO` | `your-username/stockpro-lstm` |
| `ML_BACKEND_API_KEY` | The same random secret you chose in Step 2.1 |

### 5.2 Push the ml/ directory to the Space

HF Spaces use a separate Git repo. Push the `ml/` contents to it:

```bash
cd ml

# Clone your Space repo
git clone https://huggingface.co/spaces/your-username/stockpro-ml hf-space
cd hf-space

# Copy all ml/ files into it
cp -r ../app ../scripts ../models/embeddings.json \
      ../requirements.txt ../Dockerfile .

# The .pt model is downloaded from HF Hub automatically at runtime
# so we do NOT commit it here (it would be too large)
echo "models/lstm_stock.pt" >> .gitignore
mkdir -p models
cp ../models/embeddings.json models/

git add .
git commit -m "feat: initial ML backend deploy"
git push
```

The Space will build the Docker image automatically. Watch build logs at:
`https://huggingface.co/spaces/your-username/stockpro-ml`

Build takes ~3-5 minutes. Once the Space shows **Running**, test it:

```bash
curl https://your-username-stockpro-ml.hf.space/health
# → {"status":"ok"}

curl -X POST https://your-username-stockpro-ml.hf.space/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret" \
  -d '{"symbol":"BBCA","days":7}'
```

---

## Step 6 — Configure Vercel Environment Variables

In your Vercel project dashboard → **Settings → Environment Variables**, add:

| Variable | Value | Environment |
|----------|-------|-------------|
| `ML_BACKEND_URL` | `https://your-username-stockpro-ml.hf.space` | Production |
| `ML_BACKEND_API_KEY` | The random secret from Step 2.1 | Production |

Then redeploy:

```bash
# Or just push a commit — Vercel auto-deploys
git commit --allow-empty -m "chore: trigger redeploy with ML env vars"
git push
```

---

## Step 7 — Set Up GitHub Actions Secrets

In your GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret name | Value |
|-------------|-------|
| `HF_TOKEN` | Your HF write token from Step 1.1 |
| `HF_MODEL_REPO` | `your-username/stockpro-lstm` |

`GITHUB_TOKEN` is provided automatically — no action needed.

---

## Step 8 — Trigger the First GitHub Actions Run

1. Go to your GitHub repo → **Actions** tab
2. Click **"Train LSTM Model (Weekly)"** → **Run workflow** → **Run workflow**
3. Click **"Update Stock Embeddings (Daily)"** → **Run workflow** → **Run workflow**

Watch the logs. After the training workflow completes:
- New `lstm_stock.pt` is uploaded to HF Hub
- HF Space will pick it up on the next inference request

After the embedding workflow:
- `ml/models/embeddings.json` is auto-committed to main branch
- Pull it locally: `git pull`

---

## Step 9 — Keep the HF Space in Sync (Ongoing)

The HF Space is a separate Git repo. When you update ML code, sync it:

```bash
# From the hf-space/ directory you cloned in Step 5.2
cp -r ../app ../scripts ../requirements.txt ../Dockerfile .
cp ../models/embeddings.json models/
git add .
git commit -m "update: sync ML backend"
git push
```

> **Tip:** You can automate this with a third GitHub Actions workflow
> (`.github/workflows/sync-hf-space.yml`) triggered on push to `ml/` paths.

---

## Troubleshooting

### HF Space is sleeping (cold start)
Free Spaces pause after 48h of inactivity. The Next.js forecast route sends
a `/health` warm-up ping before the real request. First call after sleep
may take 15-30s — this is normal.

To prevent sleeping: upgrade the Space to **$9/mo persistent** tier in HF Space settings.

### Model not found on HF Hub
Run Step 3 again locally. Check that `HF_TOKEN` and `HF_MODEL_REPO` are set
in your `ml/.env` file.

### GitHub Actions training fails on yfinance timeout
yfinance occasionally rate-limits. Re-run the workflow — it retries per ticker
and skips failures gracefully.

### Embeddings workflow fails to push
Ensure the workflow has `permissions: contents: write` (already set).
If your repo has branch protection on `main`, add an exception for
`github-actions[bot]` in branch protection rules.

### Similar tab shows "ML backend not configured"
`ML_BACKEND_URL` is not set in Vercel. Check Step 6.
