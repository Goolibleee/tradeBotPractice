from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_5",
    "log_return_10",
    "volatility_10",
    "atr",
    "volume_ratio",
    "fast_ma_dist",
    "slow_ma_dist",
    "trend_ma_dist",
    "body_size",
    "upper_shadow",
    "lower_shadow",
    "hour_of_day",
    "day_of_week",
]

LABEL_COLUMN = "future_direction_1"


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset CSV and return (X, y) numpy arrays."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    X = []
    y = []
    for row in rows:
        features = []
        for col in FEATURE_COLUMNS:
            val = float(row[col])
            features.append(val)
        X.append(features)
        y.append(int(row[LABEL_COLUMN]))

    return np.array(X), np.array(y)


def train_test_split_by_time(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-based split: first (1 - test_size) rows for training, rest for test."""
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "logistic_regression",
) -> Any:
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
    }


def print_confusion_matrix(y_test: np.ndarray, predictions: np.ndarray) -> None:
    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(f"  True Negatives: {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives: {cm[1, 1]}")


def save_model(model: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training(
    dataset_path: str,
    model_path: str,
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
) -> None:
    print(f"Loading dataset: {dataset_path}")
    X, y = load_dataset(dataset_path)
    print(f"Samples: {len(X)}, Features: {len(FEATURE_COLUMNS)}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split_by_time(X, y, test_size)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    print(f"Training {model_type}...")
    model = train_classifier(X_train, y_train, model_type)

    print("\nEvaluation on test set:")
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    predictions = model.predict(X_test)
    print_confusion_matrix(y_test, predictions)

    # Feature importance for Random Forest
    if hasattr(model, "feature_importances_"):
        print("\nFeature importances:")
        for name, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
            print(f"  {name}: {importance:.4f}")

    save_model(model, model_path)
    print(f"\nModel saved to: {model_path}")
