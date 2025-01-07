# Utiliser Python 3.10
FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer pip et les dépendances Python nécessaires
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.10.0 && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port 5000 utilisé par l'application
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
