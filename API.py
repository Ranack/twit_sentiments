from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
import uvicorn

# Initialisation de l'API FastAPI
app = FastAPI()

# Modèle de données pour les requêtes
class PredictionRequest(BaseModel):
    text: str

# Fonction de prédiction
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Charger le tokenizer et le modèle
        tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_roberta")
        model = TFRobertaForSequenceClassification.from_pretrained("./fine_tuned_roberta")

        # Tokenisation de l'entrée
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,  # Utilisation de la même longueur que pendant l'entraînement
            padding="max_length",
            truncation=True
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
            "confidence": float(probabilities[predicted_label])
        }

    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}
    
@app.get("/")
def read_root():
    return {"message": "API de classification de texte avec RoBERTa fine-tuné"}

# Démarrage du serveur FastAPI sur le port 8080
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Utiliser le port 8080
