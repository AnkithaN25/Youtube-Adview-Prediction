"""
Train and evaluate all five regression models on YouTube AdView data.

Models:
    1. Linear Regression
    2. Decision Tree Regressor
    3. Random Forest Regressor
    4. Support Vector Regressor (SVR)
    5. Artificial Neural Network (Keras)

Usage
-----
    python3 src/train.py
    python3 src/train.py --model rf        # train only Random Forest
    python3 src/train.py --model ann       # train only ANN
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    ANN_EPOCHS,
    ANN_HIDDEN,
    ANN_MODEL_PATH,
    DT_MODEL_PATH,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
    SCALER_PATH,
    TRAIN_CSV,
)
from src.preprocess import load_and_prepare


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: pd.DataFrame, label: str) -> dict:
    """
    Predict and print MAE, MSE, RMSE for a fitted model.

    Args:
        model:  Any fitted sklearn or Keras model with a .predict() method.
        X_test: Scaled test features.
        y_test: True target values.
        label:  Display name for the model.

    Returns:
        Dict with mae, mse, rmse.
    """
    preds = model.predict(X_test)
    mae   = metrics.mean_absolute_error(y_test, preds)
    mse   = metrics.mean_squared_error(y_test, preds)
    rmse  = np.sqrt(mse)

    print(f"\n── {label} ──")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  MSE  : {mse:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    return {"model": label, "mae": mae, "mse": mse, "rmse": rmse}


# ── Individual model trainers ─────────────────────────────────────────────────

def train_linear(X_train, y_train):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, DT_MODEL_PATH)
    print(f"Decision Tree saved → {DT_MODEL_PATH}")
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_svr(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model


def train_ann(X_train, y_train, n_features: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Dense

    model = keras.Sequential([
        Dense(ANN_HIDDEN, activation="relu", input_shape=(n_features,)),
        Dense(ANN_HIDDEN, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    model.fit(X_train, y_train, epochs=ANN_EPOCHS, verbose=0)
    model.summary()

    os.makedirs("models", exist_ok=True)
    model.save(ANN_MODEL_PATH)
    print(f"ANN saved → {ANN_MODEL_PATH}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_choice: str = "all") -> None:
    X_train, X_test, y_train, y_test, _ = load_and_prepare(
        TRAIN_CSV, is_train=True, save_scaler=True
    )

    results = []

    if model_choice in ("all", "lr"):
        lr = train_linear(X_train, y_train)
        results.append(evaluate(lr, X_test, y_test, "Linear Regression"))

    if model_choice in ("all", "dt"):
        dt = train_decision_tree(X_train, y_train)
        results.append(evaluate(dt, X_test, y_test, "Decision Tree"))

    if model_choice in ("all", "rf"):
        rf = train_random_forest(X_train, y_train)
        results.append(evaluate(rf, X_test, y_test, "Random Forest"))

    if model_choice in ("all", "svr"):
        svr = train_svr(X_train, y_train)
        results.append(evaluate(svr, X_test, y_test, "SVR"))

    if model_choice in ("all", "ann"):
        ann = train_ann(X_train, y_train, n_features=X_train.shape[1])
        results.append(evaluate(ann, X_test, y_test, "ANN"))

    if results:
        summary = pd.DataFrame(results).set_index("model")
        print("\n── Summary ──────────────────────────────")
        print(summary.to_string())
        os.makedirs("reports", exist_ok=True)
        summary.to_csv("reports/model_comparison.csv")
        print("\nSaved → reports/model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YouTube AdView prediction models.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "lr", "dt", "rf", "svr", "ann"],
        help="Which model to train (default: all)",
    )
    args = parser.parse_args()
    main(args.model)
