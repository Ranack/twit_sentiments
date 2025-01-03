import pytest
from fastapi.testclient import TestClient
from API import app 

# Initialisation du client de test
client = TestClient(app) 

# Test 1 : Vérifier que l'API retourne un message pour l'endpoint racine
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de classification de texte avec RoBERTa fine-tuné"}

# Test 2 : Vérifier que l'API retourne une prédiction valide pour une requête correcte
def test_predict_valid_text():
    payload = {"text": "I love using your app"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert isinstance(data["predicted_label"], int)
    assert isinstance(data["confidence"], float)

# Test 3 : Vérifier une prédiction négative
def test_predict_negative_text():
    payload = {"text": "I hate this app."}
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


# Test 4 : Vérifier la gestion d'une requête incorrecte (absence de champ "text")
def test_predict_missing_text():
    payload = {}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Erreur de validation (Unprocessable Entity)
    assert "detail" in response.json()


# Test 5 : Vérifier la réponse pour une requête contenant des caractères spéciaux
def test_predict_special_characters():
    payload = {"text": "@#%&*()$!"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert isinstance(data["predicted_label"], int)
    assert isinstance(data["confidence"], float)
