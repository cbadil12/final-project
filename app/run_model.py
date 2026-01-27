# app/run_model.py

# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import logging
import re

# 2. Third-party libraries
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from functools import lru_cache

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_EXT = ".joblib"
THRESHOLD = 0.5
DEFAULT_RESOLUTION = "1h"

# Defaults por task (para que no te rompa cuando task="price" y no pasas model_name)
DEFAULT_MODEL_BY_TASK = {
    "sentiment": "xgb_clf",
    "price": "rf_price",
}

# Permitir override explícito por variable de entorno (muy útil en Render)
ENV_MODEL_DIR = os.getenv("MODEL_DIR")


def _infer_project_root() -> Path:
    """
    Sube desde el directorio del archivo hasta encontrar un marcador típico de raíz de repo.
    Funciona en local, Codespaces y Render.
    """
    here = Path(__file__).resolve()
    start = here.parent  # ✅ empezar por el directorio, NO por el archivo

    markers = {"requirements.txt", "pyproject.toml", "setup.cfg", ".git", "README.md"}

    for parent in [start] + list(start.parents):
        try:
            names = {p.name for p in parent.iterdir()}
        except (PermissionError, NotADirectoryError):
            continue
        if markers & names:
            return parent

    return Path.cwd().resolve()


def _dir_has_joblibs(d: Path) -> bool:
    """True si el directorio existe y contiene al menos un .joblib"""
    return d.exists() and d.is_dir() and any(d.glob(f"*{MODEL_EXT}"))


def _candidate_model_dirs() -> list[Path]:
    """
    Candidatos donde podrían estar los modelos.
    OJO: preferimos primero rutas tipo src/models, porque en Render
    es común que la raíz real sea /opt/render/project/src.
    """
    root = _infer_project_root()

    candidates = [
        Path(ENV_MODEL_DIR).resolve() if ENV_MODEL_DIR else None,  # 1) override
        root / "src" / "models",                                   # 2) <repo>/src/models
        root / "models",                                           # 3) <repo>/models
        root / "app" / "models",                                   # 4) <repo>/app/models
        Path(__file__).resolve().parent / "models",                # 5) junto al archivo
        Path(__file__).resolve().parent.parent / "models",         # 6) un nivel arriba
        Path.cwd() / "models",                                     # 7) CWD/models
    ]

    out, seen = [], set()
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def resolve_model_dir(required: bool = True) -> Path:
    """
    Devuelve el primer directorio que EXISTA y CONTENGA *.joblib.
    Esto evita elegir carpetas vacías como /opt/render/project/models en Render.
    """
    for cand in _candidate_model_dirs():
        if _dir_has_joblibs(cand):
            return cand

    if required:
        debug = "\n".join(f"- {p} (exists={p.exists()})" for p in _candidate_model_dirs())
        raise FileNotFoundError(
            "No pude localizar una carpeta 'models' que contenga modelos (*.joblib).\n"
            "Probé estos lugares:\n"
            f"{debug}\n\n"
            "Sugerencias:\n"
            "  • En Render, define MODEL_DIR=/opt/render/project/src/models (si ahí están).\n"
            "  • Asegúrate de que los *.joblib estén commiteados (no ignorados) y se desplieguen.\n"
        )

    return _candidate_model_dirs()[0]


PROJECT_ROOT = str(_infer_project_root())
MODEL_DIR = str(resolve_model_dir(required=True))

logging.info(f"PROJECT_ROOT={PROJECT_ROOT}")
logging.info(f"MODEL_DIR={MODEL_DIR}")


# ===============================
# FEATURE LOADING
# ===============================

def load_feature_columns(task: str, model_name: str, resolution: str):
    # sentiment: feature_columns_{resolution}.json  (legacy)
    # price:     feature_columns_{model_name}_{resolution}.json (nuevo)
    if task == "sentiment":
        path = os.path.join(MODEL_DIR, f"feature_columns_{resolution}.json")
    elif task == "price":
        path = os.path.join(MODEL_DIR, f"feature_columns_{model_name}_{resolution}.json")
    else:
        raise ValueError("task must be 'sentiment' or 'price'")

    if not os.path.exists(path):
        logging.warning(f"Feature columns file not found: {path}")
        return None

    with open(path, "r") as f:
        cols = json.load(f)

    return cols if isinstance(cols, list) and cols else None


def align_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0

    extra = [c for c in out.columns if c not in feature_cols]
    if extra:
        out = out.drop(columns=extra, errors="ignore")

    return out[feature_cols]


# ===============================
# MODEL LOADING
# ===============================
_ALLOWED_RESOLUTIONS = {"1h", "4h", "24h"}


def _parse_model_filename(filename: str):
    """
    Acepta nombres como:
      - xgb_clf_1h.joblib
      - rf_price_4h.joblib
    Devuelve: (model_name, resolution) o (None, None)
    """
    if not filename.endswith(MODEL_EXT):
        return None, None

    base = filename[: -len(MODEL_EXT)]
    m = re.match(r"^(?P<model_name>.+)_(?P<res>1h|4h|24h)$", base)
    if not m:
        return None, None

    return m.group("model_name"), m.group("res")


@lru_cache(maxsize=1)
def discover_available_models():
    """
    Inspecciona MODEL_DIR y devuelve inventario de modelos disponibles.
    """
    available = {
        "sentiment": {"models": set(), "resolutions": set()},
        "price": {"models": set(), "resolutions": set()},
        "all": set(),
    }

    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"MODEL_DIR no existe: {MODEL_DIR}")

    for f in model_dir.iterdir():
        if not f.is_file():
            continue

        model_name, res = _parse_model_filename(f.name)
        if not model_name or res not in _ALLOWED_RESOLUTIONS:
            continue

        task = "price" if "price" in model_name.lower() else "sentiment"
        available[task]["models"].add(model_name)
        available[task]["resolutions"].add(res)
        available["all"].add((model_name, res))

    return available


def _validate_task_model_resolution(task: str, model_name: str, resolution: str):
    inv = discover_available_models()

    if task not in ("sentiment", "price"):
        raise ValueError("task debe ser 'sentiment' o 'price'.")

    if not inv[task]["models"]:
        raise ValueError(f"No se encontraron modelos para task='{task}' en {MODEL_DIR}")

    if resolution not in inv[task]["resolutions"]:
        raise ValueError(
            f"Resolution '{resolution}' no disponible para task='{task}'. "
            f"Disponibles: {sorted(inv[task]['resolutions'])}"
        )

    if model_name not in inv[task]["models"]:
        raise ValueError(
            f"Model '{model_name}' no disponible para task='{task}'. "
            f"Disponibles: {sorted(inv[task]['models'])}"
        )


@lru_cache(maxsize=16)
def load_model(task: str = "sentiment", resolution: str = DEFAULT_RESOLUTION, model_name: str | None = None):
    """
    Carga un modelo desde MODEL_DIR.
    """
    task = (task or "").strip().lower()
    resolution = (resolution or "").strip().lower()

    if model_name is None:
        model_name = DEFAULT_MODEL_BY_TASK.get(task, "xgb_clf")
    model_name = model_name.strip()

    _validate_task_model_resolution(task, model_name, resolution)

    model_filename = f"{model_name}_{resolution}{MODEL_EXT}"
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    logging.info(f"✅ Model loaded: {model_filename}")
    return model

# ===============================
# PREDICTION FUNCTION
# ===============================

def run_prediction(
    features_df: pd.DataFrame,
    target_ts: pd.Timestamp,
    model,
    resolution: str,
    task: str,
    model_name: str,
    drop_cols: list = None
) -> dict:

    if features_df is None or features_df.empty:
        logging.warning("Empty features DF - returning neutral fallback")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}

    target_ts = pd.to_datetime(target_ts, utc=True)

    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index, utc=True, errors="coerce")
    elif features_df.index.tz is None:
        features_df.index = features_df.index.tz_localize("UTC")

    features_df = features_df.sort_index()

    try:
        pos = features_df.index.get_indexer([target_ts], method="nearest")[0]
        row = features_df.iloc[[pos]].copy()
    except Exception as e:
        logging.error(f"Failed to select row for {target_ts}: {e}")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}

    if drop_cols:
        row = row.drop(columns=[c for c in drop_cols if c in row.columns], errors="ignore")

    feature_cols = load_feature_columns(task, model_name, resolution)
    if feature_cols:
        row = align_features(row, feature_cols)

    row = row.fillna(0.0)

    try:
        X = row.values.astype(np.float32)
        proba = model.predict_proba(X)[0]
        proba_up = float(proba[1])
        pred = 1 if proba_up >= THRESHOLD else 0
        confidence = max(proba_up, 1 - proba_up)

        logging.info(
            f"[{task.upper()} | {model_name} | {resolution}] "
            f"ts={target_ts} pred={pred} proba_up={proba_up:.3f}"
        )

        return {"prediction": pred, "proba_up": proba_up, "confidence": confidence}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}


# ===============================
# ENTRY POINT (standalone test)
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "price"], default="sentiment")
    parser.add_argument("--resolution", choices=["1h", "4h", "24h"], default="1h")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--dump_csv", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    default_input_by_task = {
        "sentiment": {
            "1h": "data/interim/aggregated_1h.csv",
            "4h": "data/interim/aggregated_4h.csv",
            "24h": "data/interim/aggregated_24h.csv",
        },
        "price": {
            "1h": "data/interim/price_features_1h.csv",
            "4h": "data/interim/price_features_4h.csv",
            "24h": "data/interim/price_features_24h.csv",
        },
    }

    inv = discover_available_models()
    task = args.task

    def _run_one(task: str, resolution: str, model_name: str, input_path: str):
        if not os.path.exists(input_path):
            logging.warning(f"Test skipped (missing input): {input_path}")
            return

        df_features = pd.read_csv(input_path, index_col=0)
        model = load_model(task=task, resolution=resolution, model_name=model_name)
        last_ts = pd.to_datetime(df_features.index[-1])

        result = run_prediction(
            df_features,
            last_ts,
            model,
            resolution=resolution,
            task=task,
            model_name=model_name,
        )

        print("\n" + "=" * 40)
        print(f"✅ TASK={task} | MODEL={model_name} | RES={resolution}")
        print(f"TS: {last_ts}")
        print(f"DIRECTION: {'UP 📈' if result['prediction'] == 1 else 'DOWN 📉'}")
        print(f"PROBA_UP: {result['proba_up']:.4f}")
        print(f"CONFIDENCE: {result['confidence']:.2%}")
        print("=" * 40 + "\n")

        if args.dump_csv:
            os.makedirs("data/processed", exist_ok=True)
            out_path = os.path.join(
                "data/processed",
                f"standalone_pred_{task}_{model_name}_{resolution}.csv"
            )
            pd.DataFrame([{
                "timestamp": last_ts,
                "task": task,
                "model": model_name,
                "resolution": resolution,
                "prediction": result["prediction"],
                "proba_up": result["proba_up"],
                "confidence": result["confidence"],
            }]).to_csv(out_path, index=False)
            logging.info(f"Saved standalone CSV: {out_path}")

    # -- ALL
    if args.all:
        for res in sorted(inv[task]["resolutions"]):
            input_path = default_input_by_task[task].get(res)
            if not input_path:
                continue
            for m in sorted(inv[task]["models"]):
                _run_one(task, res, m, input_path)
        raise SystemExit(0)

    # -- Single
    resolution = args.resolution
    model_name = args.model  # puede ser None
    input_path = args.input or default_input_by_task[task].get(resolution)

    if input_path is None:
        raise ValueError("No input path provided. Use --input <path_to_features_csv>")

    # Si no pasas modelo, usamos default por task
    if model_name is None:
        model_name = DEFAULT_MODEL_BY_TASK.get(task, "xgb_clf")

    _run_one(task, resolution, model_name, input_path)
