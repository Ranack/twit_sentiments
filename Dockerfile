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

# Mettre à jour pip avant d'installer les dépendances
RUN pip install --upgrade pip

# Installer les dépendances Python nécessaires à partir du fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape finale pour créer l'image de production
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers depuis l'étape de build
COPY --from=build /app /app

# Installer les dépendances Python supplémentaires
RUN pip install --upgrade pip

# Installer les bibliothèques nécessaires, y compris tensorflow et Streamlit
RUN pip install --no-cache-dir tensorflow==2.10.0 uvicorn streamlit fastapi

# Définir la variable d'environnement pour Azure (port 5000 pour l'API)
ENV WEBSITES_PORT=5000

# Exposer le port utilisé par l'API (5000) et Streamlit (8501)
EXPOSE 5000
EXPOSE 8501

# Créer un script d'entrée pour gérer l'exécution de l'API et Streamlit
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Commande d'entrée pour démarrer l'API et Streamlit
CMD ["/app/entrypoint.sh"]
