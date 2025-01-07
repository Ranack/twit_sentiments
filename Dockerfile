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

# Installer les dépendances
RUN pip install --upgrade pip

# Installer les bibliothèques nécessaires
RUN pip install --no-cache-dir tensorflow==2.10.0 uvicorn streamlit fastapi

# Exposer les ports utilisés par l'API et Streamlit
EXPOSE 5000
EXPOSE 8501

# Créer un script d'entrée pour gérer l'exécution de l'API et de Streamlit
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Commande d'entrée pour démarrer l'API et Streamlit
CMD ["/app/entrypoint.sh"]
