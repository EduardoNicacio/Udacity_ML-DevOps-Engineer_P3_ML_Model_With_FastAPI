"""Unit tests for the Census salary prediction API."""

import pytest
from fastapi.testclient import TestClient

from main import app

LOW_INCOME_PAYLOAD = {
    "age": 25,
    "workclass": "Private",
    "fnlgt": 226956,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Never-married",
    "occupation": "Machine-op-inspct",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

HIGH_INCOME_PAYLOAD = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}


@pytest.fixture(scope="module")
def client():
    """Start the app so the lifespan loads the real model artifacts."""
    with TestClient(app) as test_client:
        yield test_client


def test_get_root(client):
    """GET / returns a 200 with a non-empty welcome message."""
    response = client.get("/")
    body = response.json()

    assert response.status_code == 200
    assert "message" in body
    assert isinstance(body["message"], str)
    assert body["message"]


def test_post_predict_low_income(client):
    """A low-income profile is classified as 0 (<=50K)."""
    response = client.post("/predict", json=LOW_INCOME_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_post_predict_high_income(client):
    """A high-income profile is classified as 1 (>50K)."""
    response = client.post("/predict", json=HIGH_INCOME_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
