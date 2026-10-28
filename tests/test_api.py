import main
from fastapi.testclient import TestClient


client = TestClient(main.app)

VALID_PAYLOAD = {
    "quantity": 1,
    "item_total": 449.0,
    "shipping_fee": 60.18,
    "ship_city": "PASIGHAT,",
    "ship_state": "ARUNACHAL PRADESH",
    "sku": "SKU:  DN-0WDX-VYOT",
    "cod": "Not specified",
}


def test_health_check_confirms_model_is_loaded():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_status": "loaded"}


def test_predict_returns_order_status_field():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert "predicted_order_status" in response.json()


def test_predict_rejects_invalid_input():
    invalid_payload = {**VALID_PAYLOAD, "quantity": "not-an-integer"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_prediction_is_one_of_the_model_classes():
    assert main.model is not None
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    prediction = response.json()["predicted_order_status"]
    known_labels = {str(label) for label in main.model.classes_}
    assert prediction in known_labels
