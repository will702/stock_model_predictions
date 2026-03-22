import os

# Load .env if present (local dev). In GH Actions / HF Spaces, vars come from secrets directly.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY: str = os.environ.get("ML_BACKEND_API_KEY", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
MODEL_REPO: str = os.environ.get("HF_MODEL_REPO", "")  # e.g. "username/stockpro-lstm"
MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "..", "models")
