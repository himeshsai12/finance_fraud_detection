from fastapi.testclient import TestClient

from fraud_detection.service import create_app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_endpoint_returns_probability() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/score",
        json={
            "amount": 5000.0,
            "type": "TRANSFER",
            "nameOrig": "customer_a",
            "nameDest": "merchant_b",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["status"] == "ok"
