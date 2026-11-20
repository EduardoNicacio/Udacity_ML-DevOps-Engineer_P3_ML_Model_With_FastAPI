"""Unit tests for the model training and evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = ["workclass", "education"]


def make_synthetic_data(n_rows: int = 60) -> pd.DataFrame:
    """Build a small deterministic dataframe with a binary label."""
    return pd.DataFrame(
        {
            "age": list(range(20, 20 + n_rows)),
            "workclass": ["Private", "State-gov", "Self-emp-not-inc"]
            * (n_rows // 3),
            "education": ["Bachelors", "HS-grad", "Masters"] * (n_rows // 3),
            "hours-per-week": [(i % 40) + 20 for i in range(n_rows)],
            "salary": ["<=50K", ">50K"] * (n_rows // 2),
        }
    )


@pytest.fixture(scope="module")
def processed_data():
    """Process synthetic data once and reuse it across tests."""
    data = make_synthetic_data()
    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb


@pytest.fixture(scope="module")
def fitted_model(processed_data):
    """Train a model once for the tests that need a fitted classifier."""
    X_train, y_train = processed_data[0], processed_data[1]
    return train_model(X_train, y_train)


def test_process_data_training_shapes(processed_data):
    """Training mode returns a 2-D feature matrix and 1-D labels."""
    X_train, y_train, _, _, encoder, lb = processed_data
    data = make_synthetic_data()
    expected_columns = 2 + sum(
        len(categories) for categories in encoder.categories_
    )

    assert encoder is not None
    assert lb is not None
    assert X_train.shape[0] == len(data)
    assert X_train.shape[1] == expected_columns
    assert y_train.shape == (len(data),)


def test_process_data_inference_mode(processed_data):
    """Inference mode produces the same feature shape as training."""
    X_train, _, X_test, y_test = processed_data[:4]

    assert X_test.shape[0] == y_test.shape[0]
    assert X_test.shape[1] == X_train.shape[1]


def test_train_model_returns_fitted_classifier(fitted_model):
    """The trained model exposes predict and classes_ attributes."""
    assert hasattr(fitted_model, "predict")
    assert hasattr(fitted_model, "classes_")
    assert list(fitted_model.classes_) == [0, 1]


def test_inference_returns_binary_values(fitted_model, processed_data):
    """Inference returns a numpy array of binary predictions."""
    X_test = processed_data[2]
    preds = inference(fitted_model, X_test)

    assert isinstance(preds, np.ndarray)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_perfect_predictions():
    """Identical labels and predictions yield perfect metrics."""
    y = np.array([0, 1, 0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, y)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
