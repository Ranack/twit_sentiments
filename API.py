import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel


# Fonction pour charger le tokenizer et le modèle RoBERTa déjà fine-tuné
def load_roberta_model_and_tokenizer(model_dir="./fine_tuned_roberta"):
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = TFRobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
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
    tokenizer, model = load_roberta_model_and_tokenizer("./fine_tuned_roberta")
    inputs = tokenizer(request.text, return_tensors="tf", max_length=64, padding="max_length", truncation=True)
    logits = model(inputs).logits
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    predicted_label = tf.argmax(probabilities).numpy()
    return {"text": request.text, "predicted_label": int(predicted_label), "confidence": float(probabilities[predicted_label])}


if __name__ == "__main__":
    # Lancer le serveur FastAPI
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
