# src/paths.py
from pathlib import Path

PROJECT_ROOT = Path("/workspaces/final-project")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONTRACTS_DIR = DATA_DIR / "contracts"

MODELS_DIR = PROJECT_ROOT / "models"

def ensure_dirs():
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
