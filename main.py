"""FastAPI application that serves Census salary predictions."""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

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

# Model artifacts are loaded by the lifespan handler, never at import time.
ml_models: dict = {}


class CensusData(BaseModel):
    """One Census record, using the original hyphenated column names."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )

    age: int = Field(..., examples=[39])
    workclass: str = Field(..., examples=["State-gov"])
    fnlgt: int = Field(..., examples=[77516])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(alias="education-num", examples=[13])
    marital_status: str = Field(
        alias="marital-status", examples=["Never-married"]
    )
    occupation: str = Field(..., examples=["Adm-clerical"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(alias="capital-gain", examples=[2174])
    capital_loss: int = Field(alias="capital-loss", examples=[0])
    hours_per_week: int = Field(alias="hours-per-week", examples=[40])
    native_country: str = Field(
        alias="native-country", examples=["United-States"]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifacts when the application starts."""
    ml_models["model"] = load_model(
        os.path.join(MODEL_DIR, "random_forest.joblib")
    )
    ml_models["encoder"] = load_model(
        os.path.join(MODEL_DIR, "encoder.joblib")
    )
    ml_models["lb"] = load_model(os.path.join(MODEL_DIR, "lb.joblib"))
    yield
    ml_models.clear()


app = FastAPI(
    title="Census Salary Prediction API",
    description="Predicts whether a Census respondent earns more than 50K.",
    lifespan=lifespan,
)


@app.get("/")
def get_root():
    """Return a friendly welcome message."""
    return {"message": "Welcome to the Census Salary Prediction API"}


@app.post("/predict")
def predict(item: CensusData):
    """Predict the income bracket for a single Census record.

    Returns 0 for "<=50K" and 1 for ">50K".
    """
    data = pd.DataFrame([item.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=ml_models["encoder"],
        lb=ml_models["lb"],
    )
    preds = inference(ml_models["model"], X)
    return {"prediction": int(preds[0])}
