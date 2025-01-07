#!/bin/sh

# Debug : Vérification des fichiers dans le répertoire de l'application
echo "Vérification des fichiers dans /app..."
ls -l /app

# Vérification si Streamlit est installé
echo "Vérification de l'installation de Streamlit..."
which streamlit || echo "Streamlit n'est pas installé."

# Démarrage de l'API avec Uvicorn
echo "Démarrage de l'API avec Uvicorn sur le port 5000..."
uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug > /app/api_logs.txt 2>&1 &
API_PID=$!

# Attendre que l'API démarre
sleep 10

# Test de connexion sur le port 5000
echo "Test de connexion à l'API (port 5000)..."
if nc -zv 127.0.0.1 5000; then
    echo "API (port 5000) est en cours d'exécution."
else
    echo "Échec : L'API ne répond pas sur le port 5000."
    echo "Logs de l'API :"
    cat /app/api_logs.txt
    exit 1
fi

# Démarrage de Streamlit
echo "Démarrage de Streamlit sur le port 8501..."
streamlit run /app/App.py --server.port 8501 --server.headless true > /app/streamlit_logs.txt 2>&1 &
STREAMLIT_PID=$!

# Attendre que Streamlit démarre
sleep 10

# Test de connexion sur le port 8501
echo "Test de connexion à Streamlit (port 8501)..."
if nc -zv 127.0.0.1 8501; then
    echo "Streamlit (port 8501) est en cours d'exécution."
else
    echo "Échec : Streamlit ne répond pas sur le port 8501."
    echo "Logs de Streamlit :"
    cat /app/streamlit_logs.txt
    # Tuer le processus API si Streamlit échoue
    kill $API_PID
    exit 1
fi

# Garder les processus actifs
echo "Tous les services fonctionnent correctement. En attente des processus..."
wait $API_PID $STREAMLIT_PID
