#!/bin/sh

# Debug
echo "Vérification des fichiers dans /app"
ls -l /app

# Vérifier si Streamlit est installé
which streamlit || echo "Streamlit n'est pas installé."

# Démarrer l'API avec Uvicorn
echo "Démarrage de l'API avec Uvicorn..."
uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug &

# Démarrer Streamlit depuis le répertoire /app
echo "Démarrage de Streamlit..."
streamlit run /app/App.py --server.port 8501 --server.headless true && wait
