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

# Installer TensorFlow et Uvicorn pour l'exécution de l'API
RUN pip install --no-cache-dir tensorflow-cpu==2.10.0 uvicorn

# Nettoyage des fichiers inutiles
RUN rm -rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

# Exposer le port 5000 utilisé par l'application
EXPOSE 5000

# Commande pour démarrer l'application avec uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
