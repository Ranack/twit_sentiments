import pytest
import requests

def test_predict_valid_text():
    url = "http://127.0.0.1:8080/predict/"
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
    response_json = response.json()
    assert "predicted_label" in response_json
    assert "confidence" in response_json

def test_predict_valid_texti():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": "I love this product!"}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

def test_predict_negative_text():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": "I hate this product!"}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

def test_predict_empty_text():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": ""}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    # Vérification du code de statut (on s'attend à une erreur 400 pour un texte vide)
    assert response.status_code == 400, f"Expected 400, got {response.status_code} - {response.text}"

def test_predict_special_characters():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": "!@#$%^&*()"}

    # Envoi de la requête POST
    response = requests.post(url, json=data)

    # Afficher la réponse brute pour le débogage
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage

    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

