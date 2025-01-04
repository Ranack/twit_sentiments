import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf

# Initialisation de l'API FastAPI
app = FastAPI()

# Chemin du modèle
MODEL_DIR = os.path.join(os.getcwd(), "fine_tuned_roberta")

# Vérification de la présence des fichiers nécessaires
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Le dossier du modèle {MODEL_DIR} est introuvable.")
required_files = ["config.json", "tf_model.h5", "tokenizer_config.json", "vocab.json"]
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        raise FileNotFoundError(f"Fichier requis introuvable : {file}")

# Chargement du tokenizer et du modèle
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        inputs = tokenizer(
            request.text, return_tensors="tf", max_length=64, padding="max_length", truncation=True
        )
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
