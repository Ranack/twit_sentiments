import pytest
import requests

def test_predict_valid_text():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": "I love this product!"}
    
    response = requests.post(url, json=data)
    
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

def test_predict_negative_text():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": "I hate this product!"}
    
    response = requests.post(url, json=data)
    
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"

def test_predict_empty_text():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": ""}
    
    response = requests.post(url, json=data)
    
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage
    
    assert response.status_code == 400, f"Expected 400, got {response.status_code} - {response.text}"

def test_predict_special_characters():
    url = "http://127.0.0.1:8080/predict/"
    data = {"text": "!@#$%^&*()"}
    
    response = requests.post(url, json=data)
    
    print(f"Response: {response.text}")  # Afficher la réponse brute pour le débogage
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"
