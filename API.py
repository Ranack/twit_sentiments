from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import logging
import threading
import os
import time

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Définir les répertoires de cache pour les modèles (chemin local pour le développement)
CACHE_DIR = "./cache"  # Utiliser un dossier local pour stocker le cache
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TFHUB_CACHE_DIR"] = CACHE_DIR

# Création de l'application FastAPI
app = FastAPI()

# Schéma de requête attendu
class PredictionRequest(BaseModel):
    text: str

# Définir le chemin du modèle
MODEL_DIR = "./fine_tuned_roberta"

# Variables globales pour le modèle et le tokenizer
tokenizer = None
model = None

# Fonction pour charger le modèle en arrière-plan
def load_model():
    global tokenizer, model
    try:
        logging.info("Démarrage du chargement du modèle...")
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
        model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
        logging.info("Modèle et tokenizer chargés.")
        
        # Effectuer un warm-up du modèle
        sample_input = tokenizer(
            "Sample text for warm-up",
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=64,
        )
        _ = model(sample_input)
        logging.info("Le modèle a été préchauffé.")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")

# Lancer le chargement du modèle dans un thread séparé
def load_model_thread():
    threading.Thread(target=load_model, daemon=True).start()

@app.on_event("startup")
async def startup_event():
    # Charger le modèle de manière asynchrone
    load_model_thread()

# Route de santé (Azure health check)
@app.get("/")
def health_check():
    if model is None or tokenizer is None:
        logging.warning("Le modèle n'est pas encore chargé.")
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement")
    return {"status": "ok", "message": "API is up and running"}

# Endpoint pour effectuer des prédictions à la racine
@app.post("/")  # Modifié pour l'endpoint à la racine
def predict(request: PredictionRequest):
    if model is None or tokenizer is None:
        logging.warning("Le modèle n'est pas encore chargé.")
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement")

    if not request.text.strip():
        logging.error("Texte vide reçu dans la requête.")
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    try:
        logging.info(f"Requête reçue pour prédiction : {request.text}")

        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=64,
        )
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_label = tf.argmax(probabilities).numpy()

        response = {
            "text": request.text,
            "predicted_label": int(predicted_label),
            "confidence": float(probabilities[predicted_label]),
        }

        logging.info(f"Prédiction réussie : {response}")
        return response
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="127.0.0.1", port=5000, log_level="info")
