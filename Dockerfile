# Étape de build pour installer les dépendances
FROM python:3.10-alpine AS build

# Installer les dépendances système nécessaires (le paquet libmagic et d'autres outils de base)
RUN apk update && apk add --no-cache \
    git \
    libmagic \
    build-base \
    bash \
    && rm -rf /var/cache/apk/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer pip et les dépendances Python nécessaires
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Étape finale pour créer l'image de production
FROM python:3.10-alpine

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers depuis l'étape de build
COPY --from=build /app /app

# Installer TensorFlow et les autres dépendances nécessaires
RUN pip install --no-cache-dir tensorflow==2.10.0

# Nettoyage des fichiers inutiles
RUN rm -rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

# Exposer le port 5000 utilisé par l'application
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
