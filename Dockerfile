# Étape de build pour installer les dépendances
FROM python:3.10-slim AS build

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer pip et les dépendances Python nécessaires
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Étape finale pour créer l'image de production
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers depuis l'étape de build
COPY --from=build /app /app

# Installer TensorFlow, Uvicorn, Streamlit et autres dépendances
RUN pip install --no-cache-dir tensorflow-cpu==2.10.0 uvicorn streamlit

# Nettoyage des fichiers inutiles
RUN rm -rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

# Exposer les ports utilisés par l'API et Streamlit
EXPOSE 5000
EXPOSE 8501

# Commande pour démarrer l'API et l'interface Streamlit
CMD ["sh", "-c", "uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug & streamlit run app.py --server.port 8501 --server.headless true"]
