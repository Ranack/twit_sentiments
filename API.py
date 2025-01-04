from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
import os

# Configuration du répertoire du modèle
MODEL_DIR = "./fine_tuned_roberta"

# Vérification des fichiers nécessaires
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Le dossier {MODEL_DIR} est introuvable.")

required_files = ["config.json", "vocab.json", "merges.txt", "tf_model.h5", "tokenizer_config.json"]
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        raise FileNotFoundError(f"Le fichier {file} est introuvable dans {MODEL_DIR}.")

# Initialisation de l'API FastAPI
app = FastAPI()

# Modèle de données pour les requêtes
class PredictionRequest(BaseModel):
    text: str

# Chargement du modèle et du tokenizer
print("Chargement du tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
print("Tokenizer chargé avec succès.")

print("Chargement du modèle...")
model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
print("Modèle chargé avec succès.")

# Fonction de prédiction
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Tokenisation de l'entrée
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,  # Même longueur que pendant l'entraînement
            padding="max_length",
            truncation=True,
        )
        
        # Prédiction avec le modèle fine-tuné
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_label = tf.argmax(probabilities).numpy()
        
        # Retourner les résultats de la prédiction
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
