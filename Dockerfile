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

# Installer les dépendances (elles sont déjà dans requirements.txt, mais peut-être qu'il faut des versions spécifiques)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le modèle fine-tuné dans l'image
COPY fine_tuned_roberta /app/fine_tuned_roberta

# Exposer les ports utilisés par l'API
EXPOSE 5000

# Créer un script d'entrée pour gérer l'exécution de l'API et Streamlit
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Commande d'entrée pour démarrer l'API
CMD ["/app/entrypoint.sh"]
