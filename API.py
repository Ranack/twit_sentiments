from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import os

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Création de l'application FastAPI
app = FastAPI()

# Schéma de requête attendu
class PredictionRequest(BaseModel):
    text: str

# Définir le chemin du modèle
MODEL_DIR = "./fine_tuned_roberta"

# Charger le tokenizer et le modèle une seule fois au démarrage
try:
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Le répertoire du modèle '{MODEL_DIR}' est introuvable.")

    logging.info("Chargement du tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    logging.info("Chargement du modèle...")
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    logging.info("Tokenizer et modèle chargés avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle ou du tokenizer : {e}")
    raise RuntimeError(f"Erreur critique : impossible de charger le modèle ou le tokenizer ({e})")

# Route de santé (Azure health check)
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is up and running"}

# Endpoint pour les prédictions
@app.post("/predict/")
def predict(request: PredictionRequest):
    # Vérification que le texte n'est pas vide
    if not request.text.strip():
        logging.error("Texte vide reçu dans la requête.")
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    try:
        logging.info(f"Requête reçue pour prédiction : {request.text}")

        # Tokenisation du texte
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=64,
        )
        logging.debug(f"Entrée tokenisée : {inputs}")

        # Effectuer la prédiction
        outputs = model(inputs)
        if not hasattr(outputs, "logits"):
            logging.error("Les sorties du modèle ne contiennent pas de logits.")
            raise HTTPException(status_code=500, detail="Erreur : logits manquants dans les sorties du modèle.")

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


# Point d'entrée pour le déploiement (par exemple avec Docker ou Azure)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")
