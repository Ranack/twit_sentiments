from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import logging

# Configurer le logger
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Définir le tokenizer et le modèle globalement pour éviter de les charger à chaque requête
tokenizer = None
model = None

# Charger le tokenizer et le modèle une seule fois au démarrage de l'API
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_roberta")
        model = TFRobertaForSequenceClassification.from_pretrained("./fine_tuned_roberta")
        logging.debug("Model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Vérifier si le texte est vide avant de procéder
    if not request.text.strip():  # Si le texte est vide ou composé d'espaces
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logging.debug(f"Received request: {request}")  # Log de la requête reçue

    try:
        # Tokenisation du texte
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,
            padding="max_length",
            truncation=True
        )

        logging.debug(f"Tokenized inputs: {inputs}")  # Log des données tokenisées

        # Effectuer la prédiction
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_label = tf.argmax(probabilities).numpy()

        response = {
            "text": request.text,
            "predicted_label": int(predicted_label),
            "confidence": float(probabilities[predicted_label])
        }

        logging.debug(f"Response: {response}")  # Log de la réponse

        return response

    except Exception as e:
        # Log détaillé en cas d'erreur
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
