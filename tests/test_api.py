from src.api.app import app

def test_home():

    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200

def test_health():

    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200

    assert response.get_json()["status"] == "healthy"

def test_info():

    client = app.test_client()

    response = client.get("/info")

    assert response.status_code == 200

    data = response.get_json()

    assert "model" in data

def test_predict():

    client = app.test_client()

    response = client.post(
        "/predict",
        json={
            "review": "This movie was amazing!"
        }
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "sentiment" in data

    assert data["sentiment"] in [
        "positive",
        "negative"
    ]

def test_predict_empty():

    client = app.test_client()

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 400