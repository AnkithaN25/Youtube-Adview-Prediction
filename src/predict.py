"""
Run inference on test.csv using the saved Decision Tree model.

Loads the fitted scaler and Decision Tree from models/, applies them
to test.csv, and writes predictions to reports/PredictedAdview.csv.

Usage
-----
    python3 src/predict.py
    python3 src/predict.py --model dt     # Decision Tree (default)
    python3 src/predict.py --model ann    # ANN
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ANN_MODEL_PATH,
    DT_MODEL_PATH,
    PREDICTIONS_CSV,
    SCALER_PATH,
    TEST_CSV,
)
from src.preprocess import clean


def predict_decision_tree(X_scaled: np.ndarray) -> np.ndarray:
    model = joblib.load(DT_MODEL_PATH)
    print(f"Loaded Decision Tree from {DT_MODEL_PATH}")
    return model.predict(X_scaled)


def predict_ann(X_scaled: np.ndarray) -> np.ndarray:
    from tensorflow import keras
    model = keras.models.load_model(ANN_MODEL_PATH)
    print(f"Loaded ANN from {ANN_MODEL_PATH}")
    return model.predict(X_scaled).flatten()


def run_predictions(model_choice: str = "dt") -> pd.DataFrame:
    """
    Load test.csv, preprocess, run inference, save predictions.

    Args:
        model_choice: 'dt' for Decision Tree, 'ann' for ANN.

    Returns:
        DataFrame of predicted adview counts.
    """
    # Load and clean test data
    df_test = clean(pd.read_csv(TEST_CSV), is_train=False)

    # Apply the scaler fitted on training data
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(df_test)

    # Predict
    if model_choice == "dt":
        preds = predict_decision_tree(X_scaled)
    elif model_choice == "ann":
        preds = predict_ann(X_scaled)
    else:
        raise ValueError(f"Unknown model: {model_choice}. Choose 'dt' or 'ann'.")

    # Save
    os.makedirs("reports", exist_ok=True)
    result_df = pd.DataFrame(preds, columns=["predicted_adview"])
    result_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved → {PREDICTIONS_CSV}")
    print(result_df.describe())
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YouTube AdView predictions on test.csv.")
    parser.add_argument(
        "--model", default="dt", choices=["dt", "ann"],
        help="Model to use for inference: 'dt' (Decision Tree) or 'ann' (ANN). Default: dt"
    )
    args = parser.parse_args()
    run_predictions(args.model)
