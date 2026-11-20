# Script to train machine learning model.

import contextlib
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_model,
    train_model,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "census_clean.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
SLICE_PATH = os.path.join(ROOT_DIR, "slice_output.txt")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    """Run the training pipeline: split, process, train, evaluate and save."""
    data = pd.read_csv(DATA_PATH)

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Overall precision: {precision:.4f}")
    print(f"Overall recall: {recall:.4f}")
    print(f"Overall F1: {fbeta:.4f}")

    with open(SLICE_PATH, "w", encoding="utf-8") as slice_file:
        with contextlib.redirect_stdout(slice_file):
            compute_slice_metrics(
                model, encoder, lb, test, "education", cat_features
            )
    print(f"Wrote slice metrics to {SLICE_PATH}")

    save_model(model, os.path.join(MODEL_DIR, "random_forest.joblib"))
    save_model(encoder, os.path.join(MODEL_DIR, "encoder.joblib"))
    save_model(lb, os.path.join(MODEL_DIR, "lb.joblib"))
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
