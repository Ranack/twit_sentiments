import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

# Initialisation du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TextClassificationAPI")

# Définir le répertoire du modèle
MODEL_DIR = "./fine_tuned_roberta"

# Vérification que le répertoire et les fichiers nécessaires existent
if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Le répertoire {MODEL_DIR} est manquant.")

required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json']
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        raise RuntimeError(f"Fichier manquant : {file} dans {MODEL_DIR}")

# Chargement du modèle et du tokenizer
logger.info(f"Chargement du modèle et du tokenizer depuis {MODEL_DIR}...")
try:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
    logger.info("Modèle et tokenizer chargés avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
    raise RuntimeError(f"Impossible de charger le modèle ou le tokenizer. Détails : {str(e)}")

# Initialisation de l'application FastAPI
app = FastAPI()

# Modèle de données pour les requêtes
class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "API de classification de texte avec RoBERTa fine-tuné"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        logger.info(f"Requête reçue pour le texte : {request.text}")

        # Validation de l'entrée
        if not request.text.strip():
            logger.warning("Texte vide reçu dans la requête.")
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
        
        # Tokenisation
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,
            padding="max_length",
            truncation=True
        )
        logger.info("Tokenisation terminée.")

        # Prédiction
        logits = model(inputs).logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_label = tf.argmax(probabilities).numpy()

        logger.info(f"Label prédit : {predicted_label}, Confiance : {probabilities[predicted_label]}")

        return {
            "text": request.text,
            "predicted_label": int(predicted_label),
            "confidence": float(probabilities[predicted_label])
        }

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
