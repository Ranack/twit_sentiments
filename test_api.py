import pytest
from fastapi.testclient import TestClient
from API import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de classification de texte avec RoBERTa fine-tuné"}

def test_predict_valid_text():
    payload = {"text": "I love using your app"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert data["predicted_label"] in [0, 1]

def test_predict_negative_text():
    payload = {
        "text": "I hate this app. It is the worst experience I've ever had."
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert data["predicted_label"] in [0, 1]

def test_predict_empty_text():
    payload = {"text": ""}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data

def test_predict_special_characters():
    payload = {"text": "@#%&*()$!"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
