import os
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Spécifie le chemin du modèle fine-tuné
MODEL_DIR = os.path.abspath("./fine_tuned_roberta")

# Vérification des fichiers nécessaires
required_files = ["config.json", "tf_model.h5", "tokenizer_config.json", "vocab.json"]
for file in required_files:
    if not os.path.isfile(os.path.join(MODEL_DIR, file)):
        print(f"Erreur : fichier manquant {file}")
        exit(1)

# Chargement du modèle et du tokenizer
try:
    print("Chargement du tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    print("Modèle et tokenizer chargés avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou du tokenizer : {e}")
    exit(1)

# Création de l'application FastAPI
app = FastAPI()

# Modèle Pydantic pour la validation des données d'entrée
class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
async def predict(request: TextRequest):
    text = request.text

    try:
        # Tokenisation du texte
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        # Prédiction
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.numpy().argmax(axis=-1)[0]
        return {"text": text, "prediction": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

if __name__ == "__main__":
    # Lance l'API avec Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
