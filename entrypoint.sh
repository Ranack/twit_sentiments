#!/bin/sh

# Debug : Vérification des fichiers dans le répertoire de l'application
echo "Vérification des fichiers dans /app..."
ls -l /app

# Vérification si Streamlit est installé
echo "Vérification de l'installation de Streamlit..."
which streamlit > /dev/null || { echo "Streamlit n'est pas installé."; exit 1; }

# Vérification si uvicorn est installé
echo "Vérification de l'installation de uvicorn..."
which uvicorn > /dev/null || { echo "Uvicorn n'est pas installé."; exit 1; }

# Activation de l'environnement virtuel
echo "Activation de l'environnement virtuel..."
. /app/venv/bin/activate  # Remplacer 'source' par '.' pour sh

# Démarrage de l'API avec Uvicorn sur le port 5000
echo "Démarrage de l'API avec Uvicorn sur le port 5000..."
uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug > /app/api_logs.txt 2>&1 &
API_PID=$!

# Attente que l'API soit prête (temps ajustable)
echo "Attente de 120 secondes pour démarrer l'API..."
sleep 120

# Test de connexion sur le port 5000 (port de l'API)
echo "Test de connexion à l'API (port 5000)..."
if nc -zv 127.0.0.1 5000; then
    echo "API (port 5000) est en cours d'exécution."
else
    echo "Échec : L'API ne répond pas sur le port 5000."
    echo "Logs de l'API :"
    cat /app/api_logs.txt
    kill $API_PID  # Tuer le processus API si échec
    exit 1
fi

# Démarrage de Streamlit sur le port 8000
echo "Démarrage de Streamlit sur le port 8000..."
streamlit run /app/App.py --server.port 8000 --server.headless true --server.enableCORS false > /app/streamlit_logs.txt 2>&1 &
STREAMLIT_PID=$!

# Attente que Streamlit soit prêt (temps ajustable)
echo "Attente de 120 secondes pour démarrer Streamlit..."
sleep 120

# Test de connexion sur le port 8000 (port de Streamlit)
echo "Test de connexion à Streamlit (port 8000)..."
if nc -zv 127.0.0.1 8000; then
    echo "Streamlit (port 8000) est en cours d'exécution."
else
    echo "Échec : Streamlit ne répond pas sur le port 8000."
    echo "Logs de Streamlit :"
    cat /app/streamlit_logs.txt
    kill $API_PID $STREAMLIT_PID  # Tuer les processus si Streamlit échoue
    exit 1
fi

# Garder les processus actifs
echo "Tous les services fonctionnent correctement. En attente des processus..."
wait $API_PID $STREAMLIT_PID
