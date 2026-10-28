# Amazon Seller – Order Status Prediction

A small machine learning deployment project for predicting Amazon seller order status through a FastAPI endpoint. The API uses a saved scikit-learn Random Forest pipeline; it does not train or change the model.

## Current functionality

- `GET /` reports API health and whether the saved model loaded.
- `POST /predict` accepts `quantity`, `item_total`, `shipping_fee`, `ship_city`, `ship_state`, `sku`, and `cod`, then returns `predicted_order_status`.
- `/docs` provides the interactive API documentation.

The original Excel dataset is kept local and excluded from Git because it contains order-level information. The saved model file is included because the API needs it to make predictions.

## Technology

- Python
- FastAPI and Uvicorn
- pandas
- scikit-learn
- joblib

## Setup

Open a terminal in this project folder and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the API

```powershell
python -m uvicorn main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the endpoints. The root health endpoint is at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Model limitation

The dataset is imbalanced: it contains 160 delivered orders and 11 returned orders. The baseline test had 88.57% accuracy but zero recall for returned orders. The model may therefore miss returned orders and should not be treated as a reliable returned-order detector.
