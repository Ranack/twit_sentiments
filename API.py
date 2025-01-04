from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import os

# Configuration des logs
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Vérification que le texte n'est pas vide
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logging.debug(f"Received request: {request}")

        # Vérification que le répertoire du modèle existe
        model_dir = "./fine_tuned_roberta"
        if not os.path.exists(model_dir):
            logging.error(f"Model directory '{model_dir}' does not exist.")
            raise HTTPException(status_code=500, detail=f"Model directory '{model_dir}' does not exist.")

        # Charger le tokenizer et le modèle
        try:
            logging.debug(f"Loading tokenizer and model from {model_dir}...")
            tokenizer = RobertaTokenizer.from_pretrained(model_dir)
            model = TFRobertaForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model or tokenizer: {e}")

        logging.debug("Model and tokenizer successfully loaded.")

        # Tokenisation du texte
        try:
            logging.debug("Tokenizing input text...")
            inputs = tokenizer(request.text, return_tensors="tf", padding=True, truncation=True, max_length=64)
            logging.debug(f"Tokenized inputs: {inputs}")
        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
            raise HTTPException(status_code=500, detail=f"Error during tokenization: {e}")

        # Effectuer la prédiction
        try:
            logging.debug("Performing prediction...")
            outputs = model(inputs)
            if not hasattr(outputs, 'logits'):
                logging.error("Model did not return logits.")
                raise HTTPException(status_code=500, detail="Model did not return logits")

            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
            predicted_label = tf.argmax(probabilities).numpy()

            response = {
                "text": request.text,
                "predicted_label": int(predicted_label),
                "confidence": float(probabilities[predicted_label])
            }
            logging.debug(f"Prediction response: {response}")
            return response

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
