# Étape de build pour installer les dépendances
FROM python:3.10-slim AS build

# Installer les dépendances système nécessaires, y compris netcat-openbsd
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Créer un environnement virtuel pour l'application
RUN python -m venv /app/venv

# Activer l'environnement virtuel et installer les dépendances Python
RUN /app/venv/bin/pip install --upgrade pip \
    && /app/venv/bin/pip install --no-cache-dir tensorflow==2.10.0 \
    && /app/venv/bin/pip install --no-cache-dir protobuf==3.20.3 \
    && /app/venv/bin/pip install --no-cache-dir transformers \
    && /app/venv/bin/pip install --no-cache-dir uvicorn streamlit fastapi

# Étape finale pour créer l'image de production
FROM python:3.10-slim

# Installer netcat-openbsd dans l'image finale
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Créer un dossier pour le cache si nécessaire
RUN mkdir -p /app/cache

# Configurer les variables d'environnement pour que les bibliothèques de transformers et tensorflow utilisent ce cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TFHUB_CACHE_DIR=/app/cache

# Copier les fichiers depuis l'étape de build
COPY --from=build /app /app

# Ajouter l'environnement virtuel au PATH
ENV PATH="/app/venv/bin:$PATH"

# Exposer uniquement les ports 5000 et 8000
EXPOSE 5000
EXPOSE 8000

# Copier le script d'entrée et le rendre exécutable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Commande d'entrée pour démarrer l'API et Streamlit
CMD ["/app/entrypoint.sh"]
