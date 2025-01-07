#!/bin/sh

# Debug : Vérification des fichiers
echo "Vérification des fichiers dans /app"
ls -l /app

# Vérification si Streamlit est installé
which streamlit || echo "Streamlit n'est pas installé."

# Démarrage de l'API avec Uvicorn
echo "Démarrage de l'API avec Uvicorn..."
uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug &

# Attendre quelques secondes pour laisser le temps à l'API de démarrer
sleep 60

# Test de connexion sur le port 5000
echo "Test de connexion à l'API (port 5000)..."
if nc -zv 127.0.0.1 5000; then
    echo "API (port 5000) est en cours d'exécution."
else
    echo "Échec : L'API ne répond pas sur le port 5000."
    exit 1
fi

# Démarrage de Streamlit
echo "Démarrage de Streamlit..."
streamlit run /app/App.py --server.port 8501 --server.headless true &

# Attendre quelques secondes pour laisser le temps à Streamlit de démarrer
sleep 60

# Test de connexion sur le port 8501
echo "Test de connexion à Streamlit (port 8501)..."
if nc -zv 127.0.0.1 8501; then
    echo "Streamlit (port 8501) est en cours d'exécution."
else
    echo "Échec : Streamlit ne répond pas sur le port 8501."
    exit 1
fi

# Garder le script actif pour surveiller les processus
wait
