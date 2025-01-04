from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
import os

# Initialisation de l'API FastAPI
app = FastAPI()

# Chargement global du modèle
MODEL_DIR = os.path.abspath("./fine_tuned_roberta")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)

# Modèle de données pour les requêtes
class PredictionRequest(BaseModel):
    text: str

# Fonction de prédiction
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Tokenisation de l'entrée
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,
            padding="max_length",
            truncation=True,
        )

        # Prédiction
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_label = tf.argmax(probabilities).numpy()

        return {
            "text": request.text,
            "predicted_label": int(predicted_label),
            "confidence": float(probabilities[predicted_label]),
        }

    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "API de classification de texte avec RoBERTa fine-tuné"}
