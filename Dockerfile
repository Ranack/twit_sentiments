# Étape 1 : Compression des modèles (localement ou depuis un autre conteneur)
FROM python:3.9-alpine as compressor

WORKDIR /models

# Installer les outils nécessaires pour tar
RUN apk add --no-cache bash

# Copier les modèles dans l'image temporaire
COPY fine_tuned_roberta/ ./fine_tuned_roberta/

# Compresser les modèles
RUN tar -czvf fine_tuned_roberta.tar.gz fine_tuned_roberta

# Étape 2 : Image principale pour l'application
FROM python:3.9-alpine

WORKDIR /app

# Installer les dépendances système nécessaires (sans recommandations inutiles)
RUN apk add --no-cache \
    bash \
    wget \
    && rm -rf /var/cache/apk/*

# Copier et installer les dépendances Python
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de l'application
COPY . .

# Récupérer les modèles compressés depuis l'étape précédente
COPY --from=compressor /models/fine_tuned_roberta.tar.gz ./models/

# Décompresser les modèles dans le conteneur final
RUN mkdir -p ./models/fine_tuned_roberta && \
    tar -xzvf ./models/fine_tuned_roberta.tar.gz -C ./models/ && \
    rm ./models/fine_tuned_roberta.tar.gz

# Exposer le port et démarrer l'application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
