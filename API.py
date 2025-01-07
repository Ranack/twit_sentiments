from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import os
import threading

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Créer un dossier pour le cache si nécessaire
CACHE_DIR = "/app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TFHUB_CACHE_DIR"] = CACHE_DIR

# Création de l'application FastAPI
app = FastAPI()

# Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accepter les requêtes depuis n'importe quelle origine
    allow_credentials=True,
    allow_methods=["*"],  # Accepter toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Accepter tous les headers
)

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
        sample_input = tokenizer("Sample text for warm-up", return_tensors="tf", padding=True, truncation=True, max_length=64)
        _ = model(sample_input)
        logging.info("Le modèle a été préchauffé.")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")

# Fonction de pré-chargement du modèle dans un thread séparé
def load_model_thread():
    threading.Thread(target=load_model, daemon=True).start()

# Route de santé (Azure health check)
@app.get("/")
def health_check():
    if model is None or tokenizer is None:
        logging.warning("Le modèle n'est pas encore chargé.")
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement")
    return {"status": "ok", "message": "API is up and running"}

# Endpoint pour les prédictions
@app.post("/predict/")
def predict(request: PredictionRequest):
    if model is None or tokenizer is None:
        logging.warning("Le modèle n'est pas encore chargé.")
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement")

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
@app.on_event("startup")
async def startup_event():
    # Charger le modèle de manière asynchrone en arrière-plan
    load_model_thread()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=5000, log_level="info")
