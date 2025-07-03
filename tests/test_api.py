import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_hitter():
    # Use dummy data matching HitterInput schema
    data = {"feature1": 0.5, "feature2": 1.2}
    response = client.post("/predict/hitter", json=data)
    assert response.status_code == 200
    assert "class" in response.json() or "error" in response.json()

def test_predict_pitcher():
    # Use dummy data matching PitcherInput schema
    data = {"feature1": 0.7, "feature2": 2.3}
    response = client.post("/predict/pitcher", json=data)
    assert response.status_code == 200
    assert "class" in response.json() or "error" in response.json() 