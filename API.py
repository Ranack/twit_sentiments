@app.post("/predict/")
def predict(request: PredictionRequest):
    # Vérification que le texte n'est pas vide
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logging.debug(f"Received request: {request}")

        # Vérification que le répertoire du modèle est valide
        import os
        if not os.path.exists("./fine_tuned_roberta"):
            logging.error("Model directory './fine_tuned_roberta' does not exist.")
            raise HTTPException(status_code=500, detail="Model directory does not exist.")

        # Charger le tokenizer et le modèle
        try:
            tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_roberta")
            model = TFRobertaForSequenceClassification.from_pretrained("./fine_tuned_roberta")
        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model or tokenizer: {e}")

        logging.debug("Model and tokenizer loaded.")

        # Tokenisation du texte
        inputs = tokenizer(request.text, return_tensors="tf", padding=True, truncation=True, max_length=64)

        logging.debug(f"Tokenized inputs: {inputs}")

        # Effectuer la prédiction
        outputs = model(inputs)

        if 'logits' not in outputs:
            logging.error("Model did not return logits.")
            raise HTTPException(status_code=500, detail="Model did not return logits")

        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_label = tf.argmax(probabilities).numpy()

        response = {
            "text": request.text,
            "predicted_label": int(predicted_label),
            "confidence": float(probabilities[predicted_label])
        }

        logging.debug(f"Prediction response: {response}")
        return response

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
