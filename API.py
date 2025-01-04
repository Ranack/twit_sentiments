from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

# Initialisation de l'application FastAPI
app = FastAPI()

# Classe de modèle pour valider la structure de la requête
class PredictionRequest(BaseModel):
    text: str

# Route de prédiction
@app.post("/predict/")
def predict(request: PredictionRequest):
    print(f"Received request: {request.text}")  # Log du texte reçu

    try:
        # Charger le tokenizer et le modèle pré-entraîné (téléchargé ou local)
        print("Loading tokenizer and model...")
        tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_roberta")
        model = TFRobertaForSequenceClassification.from_pretrained("./fine_tuned_roberta")

        # Traitement du texte d'entrée
        print("Tokenizing input text...")
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,
            padding="max_length",
            truncation=True
        )

        print(f"Tokenized inputs: {inputs}")  # Log des entrées tokenisées

        # Effectuer la prédiction
        print("Making prediction...")
        outputs = model(inputs)
        logits = outputs.logits

        # Calcul des probabilités via softmax
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]

        # Prédiction et confiance associée
        predicted_label = tf.argmax(probabilities).numpy()
        confidence = float(probabilities[predicted_label])

        response = {
            "text": request.text,
            "predicted_label": int(predicted_label),
            "confidence": confidence
        }

        print(f"Response: {response}")  # Log de la réponse envoyée

        return response
    
    except Exception as e:
        # Log détaillé de l'erreur
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
