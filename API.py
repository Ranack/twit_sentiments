import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging


# Initialisation du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction pour charger le tokenizer et le modèle RoBERTa déjà fine-tuné
def load_roberta_model_and_tokenizer(model_dir="./fine_tuned_roberta"):
    logger.info(f"Chargement du modèle et du tokenizer depuis {model_dir}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = TFRobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    logger.info("Modèle et tokenizer chargés avec succès.")
    return tokenizer, model


# FastAPI pour le modèle fine-tuné
app = FastAPI()

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
        if not request.text:
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
        
        tokenizer, model = load_roberta_model_and_tokenizer("./fine_tuned_roberta")
        
        # Tokenisation
        inputs = tokenizer(request.text, return_tensors="tf", max_length=64, padding="max_length",
            truncation=True)
        logger.info("Tokenisation terminée.")
        
        # Vérification des entrées tokenisées
        if inputs.get('input_ids') is None:
            raise HTTPException(status_code=500, detail="Erreur lors de la tokenisation.")
        
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
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
