# Étape de build pour installer les dépendances
FROM python:3.10-slim AS build

# Installer les dépendances système nécessaires, y compris curl, netcat-openbsd, libstdc++ et libgcc
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    bash \
    libmagic-dev \
    libxml2-dev \
    libxslt-dev \
    libgcc1 \
    libstdc++6 \
    libc6-arm64-cross \
    libsndfile1 && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Créer un environnement virtuel pour l'application
RUN python -m venv /app/venv

# Activer l'environnement virtuel et installer les dépendances depuis le fichier requirements.txt
RUN /app/venv/bin/pip install --upgrade pip
COPY requirements.txt /app/requirements.txt

# Installer les dépendances du requirements.txt (assure-toi que tqdm, streamlit, et uvicorn sont inclus dans requirements.txt)
RUN /app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Étape finale pour créer l'image de production basée sur l'image slim
FROM python:3.10-slim

# Installer netcat-openbsd, curl, libstdc++ et libgcc dans l'image slim finale
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    libmagic-dev \
    libgcc1 \
    libstdc++6 && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Créer un dossier pour le cache si nécessaire
RUN mkdir -p /app/cache

# Configurer les variables d'environnement pour que les bibliothèques de transformers et tensorflow utilisent ce cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TFHUB_CACHE_DIR=/app/cache

# Copier uniquement les fichiers nécessaires depuis l'étape de build
COPY --from=build /app /app

# Ajouter l'environnement virtuel au PATH
ENV PATH="/app/venv/bin:$PATH"

# Exposer uniquement les ports 5000 et 8000
EXPOSE 5000
EXPOSE 8000

# Copier le script d'entrée et le rendre exécutable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Nettoyer les fichiers de cache et les fichiers temporaires pour alléger l'image
RUN rm -rf /root/.cache/pip/* /app/requirements.txt /app/venv/lib/python3.10/site-packages/*.dist-info

# Commande d'entrée pour démarrer l'API et Streamlit
CMD ["/app/entrypoint.sh"]
