import pytest
import requests

def test_predict_valid_text():
    url = "http://127.0.0.1:5000/predict/"  # Assurez-vous que le port est 5000
    data = {"text": "I love this product!"}

    headers = {
        'Content-Type': 'application/json'  # Assurez-vous que le type de contenu est JSON
    }

    # Envoi de la requête POST
    response = requests.post(url, json=data, headers=headers)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")

    # Vérification du code de statut
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

    # Vérification de la réponse JSON
    response_json = response.json()
    assert "predicted_label" in response_json, "Response missing 'predicted_label'"
    assert "confidence" in response_json, "Response missing 'confidence'"

def test_predict_negative_text():
    url = "http://127.0.0.1:5000/predict/"  # Assurez-vous que le port est 5000
    data = {"text": "I hate this product!"}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    # Vérification du code de statut
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

    # Vérification de la réponse JSON
    response_json = response.json()
    assert "predicted_label" in response_json, "Response missing 'predicted_label'"
    assert "confidence" in response_json, "Response missing 'confidence'"

def test_predict_empty_text():
    url = "http://127.0.0.1:5000/predict/"  # Assurez-vous que le port est 5000
    data = {"text": ""}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    # Vérification du code de statut (on s'attend à une erreur 400 pour un texte vide)
    assert response.status_code == 400, f"Expected 400, got {response.status_code} - {response.text}"

    # Vérification du message d'erreur dans la réponse JSON
    response_json = response.json()
    assert "detail" in response_json, "Error response missing 'detail'"
    assert response_json["detail"] == "Le texte ne peut pas être vide.", f"Expected error message 'Le texte ne peut pas être vide.', got {response_json['detail']}"

def test_predict_special_characters():
    url = "http://127.0.0.1:5000/predict/"  # Assurez-vous que le port est 5000
    data = {"text": "!@#$%^&*()"}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    # Vérification du code de statut
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

    # Vérification de la réponse JSON
    response_json = response.json()
    assert "predicted_label" in response_json, "Response missing 'predicted_label'"
    assert "confidence" in response_json, "Response missing 'confidence'"
