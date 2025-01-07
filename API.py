from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import os

# Configuration des logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Création de l'application FastAPI
app = FastAPI()

# Schéma de requête attendu
class PredictionRequest(BaseModel):
    text: str

# Définir le chemin du modèle
MODEL_DIR = "./fine_tuned_roberta"

# Charger le tokenizer et le modèle une seule fois au démarrage
try:
    logging.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    logging.info("Loading model...")
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    logging.info("Tokenizer and model loaded successfully.")
except Exception as e:
    logging.error(f"Error during loading: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading model or tokenizer: {e}")

# Endpoint pour les prédictions
@app.post("/predict/")
def predict(request: PredictionRequest):
    # Vérification que le texte n'est pas vide
    if not request.text.strip():
        logging.error("Empty text received in request.")
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logging.info(f"Received request for prediction: {request.text}")

        # Tokenisation du texte
        try:
            logging.info("Tokenizing input text...")
            inputs = tokenizer(request.text, return_tensors="tf", padding=True, truncation=True, max_length=64)
            logging.debug(f"Tokenized inputs: {inputs}")
        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
            raise HTTPException(status_code=500, detail=f"Error during tokenization: {e}")

        # Effectuer la prédiction
        try:
            logging.info("Performing prediction...")
            outputs = model(inputs)

            if not hasattr(outputs, 'logits'):
                logging.error("Model output does not contain logits.")
                raise HTTPException(status_code=500, detail="Model output does not contain logits")

            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
            predicted_label = tf.argmax(probabilities).numpy()

            response = {
                "text": request.text,
                "predicted_label": int(predicted_label),
                "confidence": float(probabilities[predicted_label])
            }

            logging.info(f"Prediction successful: {response}")
            return response
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
