#!/bin/sh

# Activer un mode plus verbeux pour capturer les erreurs dans chaque étape
set -e  # Arrêter le script si une commande échoue
set -x  # Affiche chaque commande exécutée

# Vérification si uvicorn est installé
echo "Vérification de l'installation de uvicorn..."
which uvicorn > /dev/null || { echo "Uvicorn n'est pas installé."; exit 1; }

# Vérification si tqdm est installé
echo "Vérification de l'installation de tqdm..."
python -m pip show tqdm || { echo "tqdm n'est pas installé. Installation en cours..."; /app/venv/bin/pip install tqdm; }

# Activation de l'environnement virtuel
echo "Activation de l'environnement virtuel..."
. /app/venv/bin/activate  # Remplacer 'source' par '.' pour sh

# Vérification si l'environnement virtuel est activé
if [ -z "$VIRTUAL_ENV" ]; then
    echo "L'environnement virtuel n'est pas activé."
    exit 1
fi

# Démarrage de l'API avec Uvicorn sur le port 5000
echo "Démarrage de l'API avec Uvicorn sur le port 5000..."
uvicorn API:app --host 0.0.0.0 --port 5000 --log-level debug > /app/api_logs.txt 2>&1 &
API_PID=$!

# Fonction pour vérifier si l'API est prête sur le port 5000
check_api() {
  echo "Tentatives pour vérifier si l'API est prête sur le port 5000..."
  for i in {1..20}; do
      sleep 60
      if nc -zv 127.0.0.1 5000; then
          echo "L'API est prête !"
          return 0
      fi
      echo "Tentative $i : L'API n'est pas encore prête, nouvelle tentative dans 60 secondes..."
      sleep 60
  done
  echo "Échec : L'API ne répond pas sur le port 5000."
  echo "Logs de l'API :"
  cat /app/api_logs.txt
  return 1
}

# Lancer la vérification de l'API
check_api

# Garder le processus actif
echo "Le service API fonctionne correctement. En attente du processus..."

# Effectuer une requête curl pour tester l'API
echo "Envoi d'une requête de test à l'API..."
curl -X POST http://127.0.0.1:5000/ -H "Content-Type: application/json" -d '{"text": "I love it."}'

# Attendre que le processus reste actif
wait $API_PID
