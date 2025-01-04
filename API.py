from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(request: PredictionRequest):
    print(f"Received request: {request.text}")  # Log du texte reçu

    try:
        # Charger le tokenizer et le modèle
        tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_roberta")
        model = TFRobertaForSequenceClassification.from_pretrained("./fine_tuned_roberta")

        # Traitement du texte d'entrée
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            max_length=64,
            padding="max_length",
            truncation=True
        )

        print(f"Tokenized inputs: {inputs}")  # Log des entrées tokenisées

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

        print(f"Response: {response}")  # Log de la réponse envoyée

        return response
    
    except Exception as e:
        # Log détaillé de l'erreur
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
