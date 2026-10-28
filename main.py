from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


MODEL_PATH = Path(__file__).resolve().parent / "order_status_model.pkl"
FEATURE_COLUMNS = [
    "quantity",
    "item_total",
    "shipping_fee",
    "ship_city",
    "ship_state",
    "sku",
    "cod",
]

app = FastAPI(title="Amazon Order Status Prediction API")
model: Any = None
model_load_error: str | None = None


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: int
    item_total: float
    shipping_fee: float
    ship_city: str
    ship_state: str
    sku: str
    cod: str


def load_model() -> None:
    """Load the saved pipeline from beside this file, not the current directory."""
    global model, model_load_error
    try:
        model = joblib.load(MODEL_PATH)
        model_load_error = None
    except Exception as exc:
        model = None
        model_load_error = f"Could not load model at {MODEL_PATH}: {exc}"


load_model()


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "model_status": "loaded" if model is not None else "unavailable",
    }


@app.post("/predict")
def predict_order_status(payload: PredictionInput) -> dict[str, str]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model is unavailable. {model_load_error}",
        )

    try:
        input_data = pd.DataFrame(
            [[getattr(payload, column) for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )
        prediction = model.predict(input_data)
        return {"predicted_order_status": str(prediction[0])}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"The model could not make a prediction: {exc}",
        ) from exc
