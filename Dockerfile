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

# Créer un environnement virtuel et y installer les dépendances
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /app/venv/bin/pip install --no-cache-dir tensorflow==2.10.0 uvicorn streamlit fastapi

# Étape finale pour créer l'image de production
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application depuis l'étape de build
COPY --from=build /app /app

# Définir les variables d'environnement pour Azure et Streamlit
ENV WEBSITES_PORT=5000
ENV STREAMLIT_SERVER_PORT=8501

# Ajouter l'environnement virtuel au PATH
ENV PATH="/app/venv/bin:$PATH"

# Exposer les ports utilisés par l'API (5000) et Streamlit (8501)
EXPOSE 5000
EXPOSE 8501

# Copier le script d'entrée et le rendre exécutable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Commande d'entrée pour démarrer l'API et Streamlit
CMD ["/app/entrypoint.sh"]
