# Étape 1 : Construction et installation des dépendances
FROM python:3.12-slim AS builder

# Installer les dépendances système strictement nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement requirements.txt pour profiter du cache Docker
COPY requirements.txt .

# Installer les dépendances dans un répertoire isolé
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Étape 2 : Image finale minimaliste
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer uniquement les bibliothèques système nécessaires pour l'exécution
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires pour l'application
COPY API.py .
COPY fine_tuned_roberta ./fine_tuned_roberta
COPY --from=builder /install /usr/local

# Vérifier que les binaires des dépendances installées sont dans le PATH
ENV PATH="/usr/local/bin:$PATH"

# Limiter l'usage des threads et de la mémoire pour TensorFlow
# TF_CPP_MIN_LOG_LEVEL : Supprime les logs inutiles de TensorFlow
# TF_NUM_INTEROP_THREADS : Limite les threads pour les opérations parallèles
# TF_NUM_INTRAOP_THREADS : Limite les threads internes pour TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTEROP_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2

# Configuration de TensorFlow pour limiter l'usage de la mémoire
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Exposer le port pour FastAPI
EXPOSE 5000

# Commande par défaut pour démarrer l'application
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
