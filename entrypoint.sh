#!/bin/sh

# Debug
echo "Vérification des fichiers dans /app"
ls -l /app

# Démarrer l'API avec Uvicorn
echo "Démarrage de l'API avec Uvicorn..."
uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug &

# Attente avant de démarrer Streamlit pour s'assurer que l'API est prête
echo "Attente de 5 secondes pour que l'API démarre..."
sleep 5

# Démarrer Streamlit depuis le répertoire /app
echo "Démarrage de Streamlit..."
streamlit run /app/App.py --server.port 8501 --server.headless true && wait
