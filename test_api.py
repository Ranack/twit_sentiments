import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from API import app  # Assurez-vous que le fichier principal s'appelle API.py et contient l'application FastAPI

# Initialisation du client de test
client = TestClient(app)

# Mock de MLflow pour éviter la connexion pendant les tests
@patch("mlflow.client.MlflowClient.get_experiment_by_name")
def test_root_endpoint(mock_get_experiment_by_name):
    # Configure le mock pour retourner une réponse simulée
    mock_get_experiment_by_name.return_value = {"experiment_id": "123"}

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de classification de texte avec RoBERTa fine-tuné"}

@patch("mlflow.client.MlflowClient.get_experiment_by_name")
def test_predict_valid_text(mock_get_experiment_by_name):
    # Configure le mock pour retourner une réponse simulée
    mock_get_experiment_by_name.return_value = {"experiment_id": "123"}

    payload = {"text": "I love using your app"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert isinstance(data["predicted_label"], int)
    assert isinstance(data["confidence"], float)

@patch("mlflow.client.MlflowClient.get_experiment_by_name")
def test_predict_negative_text(mock_get_experiment_by_name):
    # Configure le mock pour retourner une réponse simulée
    mock_get_experiment_by_name.return_value = {"experiment_id": "123"}

    payload = {"text": "I hate this app. It is the worst experience I've ever had."}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert isinstance(data["predicted_label"], int)
    assert isinstance(data["confidence"], float)
    # Supposons que la classe négative soit représentée par le label 0
    assert data["predicted_label"] == 0
    assert data["confidence"] > 0.5  # Une confiance raisonnable pour une prédiction négative

@patch("mlflow.client.MlflowClient.get_experiment_by_name")
def test_predict_missing_text(mock_get_experiment_by_name):
    # Configure le mock pour retourner une réponse simulée
    mock_get_experiment_by_name.return_value = {"experiment_id": "123"}

    payload = {}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Erreur de validation (Unprocessable Entity)
    assert "detail" in response.json()

@patch("mlflow.client.MlflowClient.get_experiment_by_name")
def test_predict_special_characters(mock_get_experiment_by_name):
    # Configure le mock pour retourner une réponse simulée
    mock_get_experiment_by_name.return_value = {"experiment_id": "123"}

    payload = {"text": "@#%&*()$!"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert isinstance(data["predicted_label"], int)
    assert isinstance(data["confidence"], float)
