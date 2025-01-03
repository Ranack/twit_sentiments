from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.tensorflow
import os

# Configuration de MLFlow
if not os.getenv("IGNORE_MLFLOW"):  # N'initialisez MLFlow que si l'environnement ne demande pas son ignoré
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("Fine_tuning_RoBERTa_Optimized")


# Fonction pour charger et préparer les données
def load_and_preprocess_data(file_path):
    print("Chargement des données...")
    columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    data = pd.read_csv(file_path, encoding='latin1', names=columns)

    print("Prétraitement des données...")
    data = data[['sentiment', 'text']]
    data['sentiment'] = data['sentiment'].replace({0: 0, 4: 1})
    data = data.sample(frac=0.8, random_state=42)  # Échantillonner 80% des données pour accélérer
    X = data['text'].values
    y = data['sentiment'].values

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# Fonction pour charger le tokenizer RoBERTa
def load_tokenizer():
    print("Chargement du tokenizer RoBERTa...")
    return RobertaTokenizer.from_pretrained("roberta-base")


# Fonction pour encoder les textes
def encode_texts(texts, tokenizer, max_len):
    return tokenizer(
        list(texts),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )


# Fonction pour préparer les datasets TensorFlow
def prepare_datasets(train_encodings, test_encodings, y_train, y_test, batch_size):
    print("Préparation des datasets TensorFlow...")
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)) \
        .shuffle(10000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(batch_size)
    return train_dataset, test_dataset


# Fonction pour charger le modèle RoBERTa
def load_model():
    print("Chargement du modèle RoBERTa...")
    model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Geler les couches inférieures
    for layer in model.roberta.encoder.layer[:10]:
        layer.trainable = False
    
    return model


# Fonction pour compiler le modèle
def compile_model(model, learning_rate):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Fonction pour entraîner le modèle
def train_model(model, train_dataset, test_dataset, epochs):
    print("Entraînement du modèle...")
    return model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs
    )


# Fonction pour évaluer le modèle
def evaluate_model(model, test_dataset):
    print("Évaluation du modèle...")
    results = model.evaluate(test_dataset)
    return results[0], results[1]


# Fonction pour sauvegarder le modèle et le tokenizer
def save_model_and_tokenizer(model, tokenizer, model_dir):
    print("Sauvegarde du modèle...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


# Fonction pour enregistrer les hyperparamètres et les métriques dans MLFlow
def log_metrics_in_mlflow(run, epochs, batch_size, learning_rate, max_len, test_loss, test_accuracy, model_dir):
    if not os.getenv("IGNORE_MLFLOW"):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_len", max_len)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=None,
            conda_env=None,
            code_path=None,
            loader_module="transformers",
            data_path=model_dir
        )


# Fonction pour afficher et sauvegarder les graphes de performance
def plot_and_save_graphs(history):
    print("Affichage des graphes de performance...")
    plt.figure(figsize=(12, 6))

    # Graphe de la perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Graphe de la précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("performance_graphs.png")
    plt.show()


# FastAPI pour le modèle fine-tuné
app = FastAPI()

class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "API de classification de texte avec RoBERTa fine-tuné"}


@app.post("/predict/")
def predict(request: PredictionRequest):
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


# Fonction principale pour orchestrer l'entraînement et la sauvegarde du modèle
def main():
    file_path = 'training.1600000.processed.noemoticon.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    tokenizer = load_tokenizer()
    max_len = 64
    train_encodings = encode_texts(X_train, tokenizer, max_len)
    test_encodings = encode_texts(X_test, tokenizer, max_len)
    
    batch_size = 16
    train_dataset, test_dataset = prepare_datasets(train_encodings, test_encodings, y_train, y_test, batch_size)
    
    model = load_model()
    epochs = 4
    learning_rate = 2e-4
    compile_model(model, learning_rate)
    
    with mlflow.start_run(run_name="Fine-tuning_RoBERTa_Optimized") as run:
        history = train_model(model, train_dataset, test_dataset, epochs)
        test_loss, test_accuracy = evaluate_model(model, test_dataset)
        model_dir = "./fine_tuned_roberta"
        save_model_and_tokenizer(model, tokenizer, model_dir)
        log_metrics_in_mlflow(run, epochs, batch_size, learning_rate, max_len, test_loss, test_accuracy, model_dir)
        plot_and_save_graphs(history)


if __name__ == "__main__":
    main()
