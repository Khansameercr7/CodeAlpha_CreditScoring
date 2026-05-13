"""
joblib model persistence
"""
from __future__ import annotations
import os, joblib
from src.utils.logger import get_logger

log = get_logger("utils.model_io")

def save_all(models: dict, scaler, feature_names: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(out_dir, fname))
        log.info(f"Saved: {os.path.join(out_dir, fname)}")
    joblib.dump(scaler,       os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_names.pkl"))
    log.info(f"Saved: scaler.pkl + feature_names.pkl")

def load_all(out_dir: str) -> tuple[dict, object, list]:
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"'{out_dir}' not found. Run main.py first.")
    models = {}
    for f in os.listdir(out_dir):
        if f.endswith(".pkl") and f not in ("scaler.pkl","feature_names.pkl"):
            name = (f.replace(".pkl","").replace("_"," ").title()
                     .replace("Xgboost","XGBoost"))
            models[name] = joblib.load(os.path.join(out_dir, f))
            log.info(f"Loaded: {f}")
    scaler        = joblib.load(os.path.join(out_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(out_dir, "feature_names.pkl"))
    return models, scaler, feature_names
