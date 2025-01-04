from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

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

        # Charger le tokenizer et le modèle
        tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_roberta")
        model = TFRobertaForSequenceClassification.from_pretrained("./fine_tuned_roberta")

        logging.debug("Model and tokenizer loaded.")

        # Tokenisation du texte
        inputs = tokenizer(request.text, return_tensors="tf", padding=True, truncation=True, max_length=64)

        # Effectuer la prédiction
        logging.debug(f"Tokenized inputs: {inputs}")

        outputs = model(inputs)

        if 'logits' not in outputs:
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
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
