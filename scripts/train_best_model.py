from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
# Ensure repo root is on sys.path when running as a script.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from model_utils import DEFAULT_SIGMA, extract_features, preprocess_flux
DATA_DIR = ROOT_DIR / "Data"
MODEL_DIR = ROOT_DIR / "Models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"

BEST_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.03230689460872876,
    "min_child_weight": 1,
    "scale_pos_weight": 2.39098817091045,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}


def main() -> None:
    data_path = DATA_DIR / "exo_all_combined.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    data = pd.read_csv(data_path)

    label_map = {2: 1, 1: 0}
    data["LABEL"] = data["LABEL"].map(label_map)
    if data["LABEL"].isna().any():
        raise ValueError("Unexpected label values found in dataset.")

    flux_cols = [c for c in data.columns if c != "LABEL"]
    flux_matrix = data[flux_cols].values

    processed = preprocess_flux(flux_matrix, sigma=DEFAULT_SIGMA)
    features = extract_features(processed)
    feature_names = list(features.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, data["LABEL"])

    model = XGBClassifier(**BEST_XGB_PARAMS)
    model.fit(X_res, y_res)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "sigma": DEFAULT_SIGMA,
        "n_points": len(flux_cols),
        "label_map": label_map,
        "model_name": "xgb_optuna_best",
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
