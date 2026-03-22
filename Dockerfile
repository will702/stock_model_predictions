# Hugging Face Spaces Docker template
# Port 7860 is required by HF Spaces
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (smaller image, sufficient for inference)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create models dir (pre-trained model should be committed or downloaded here)
RUN mkdir -p models

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
