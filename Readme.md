# 🇬🇧Projet Text Classification with RoBERTa Fine-Tuned


This project implements a simple user interface via **Streamlit** to interact with a text classification model based on **RoBERTa** fine-tuned. The application can analyze the main emotion (negative or positive) of texts and report errors via **Azure Application Insights**.

## Features

- Text classification**: Enter a text in English to obtain a predictive label and a confidence probability.
- Error reporting**: Send prediction information to **Azure Application Insights** when needed.
- Centralized logging**: Integration with **Azure Application Insights** to collect and analyze error logs.

## Prerequisites

Before getting started, you need to have the following installed:

- Python 3.8 or higher**.
- The **RoBERTa fine-tuned** template.
- A valid instrumentation key for **Azure Application Insights**.
- A **API Backend** accessible at URL `http://127.0.0.1:5000/` (or other configured URL).

## Installation

1. Clone this repository by running the following command:

   git clone <https://github.com/Ranack/twit_sentiments
   cd twit_sentiments 

2.  Install the dependencies with pip:

    
    `pip install -r requirements.txt` 
    
3.  Ensure that the Backend API is running correctly at the following address: `http://127.0.0.1:5000/`.
    

## Configuration

1.  Add your **Azure Application Insights** instrumentation key to the `app.py` file:
    

    
    `INSTRUMENTATION_KEY = “your instrumentation-key”`. 
    
2.  If necessary, update the API URL in `app.py`:

    
    `API_URL = “http://127.0.0.1:5000/”` 
    
    ## Use

### Local

1.  Run the Streamlit application by executing the following command:
    

    
    `streamlit run app.py` 
    
2.  Open your browser and access the user interface at the following address:
    

    
    http://localhost:8501 
    
3.  Enter text in the box provided and click on **Predict** to obtain the associated label and probability.
    
4.  To report an error, click on **Report an error** after making a prediction.

### Online

The API is also available online. You can send **POST** requests with the text to be classified using `curl`.

Here's an example of a curl command to interact with the API online:


`curl -X POST https://twisentiment-v2.azurewebsites.net/ \
    -H “Content-Type: application/json” \
    -d '{“text”: “I love the new features!”}'` `curl 

- **URL** : [https://twisentiment-v2.azurewebsites.net/](https://twisentiment-v2.azurewebsites.net/)
- HTTP method** : POST
- **Headers** : `Content-Type: application/json`
- Data**: JSON with text to be classified, e.g. `{“text”: “I love the new features!”}`
#### Sample response

The response returned by the API could look like this:



`{
  “label": ‘1’,
  “confidence": 0.98
}` 

This means that the text has been classified as positive (1) (négative is 0) with a probability of 98%.

## Project structure

Here's the project structure and a brief description of the files :

├── .github/
│ └── workflows/
│ └── python-app.yml # Configuring GitHub Actions for CI/CD
├── fine_tuned_roberta/
│ └── pytorch_model.bin # RoBERTa fine-tuned model (stored via Git LFS)
├── .dockerignore # List of files to ignore when creating the Docker image
├── .gitattributes # Definition of attributes for Git LFS
├── .gitignore # List of files to be ignored by Git
├── API.py # Backend logic for classification API
├── App.py # Graphical interface to Streamlit (local)
├── Dockerfile # Definition of Docker image for deployment
├── requirements.txt # List of Python dependencies
├── test_api.py # Unit tests for backend API
└── README.md # Project documentation` 

## Integration with Azure Application Insights

Error reporting logs are sent to Azure Application Insights. Be sure to check the following:

- The instrumentation key is correct.
- The data sent respects the JSON format expected by Azure Application Insights.

# 🇫🇷Projet de Classification de Texte avec RoBERTa Fine-Tuné

Ce projet implémente une interface utilisateur simple via **Streamlit** pour interagir avec un modèle de classification de texte basé sur **RoBERTa** fine-tuné. L'application permet d'analyser l'émotion principale (négative ou positive) des textes et de signaler des erreurs via **Azure Application Insights**.

## Fonctionnalités

- **Classification de texte** : Saisissez un texte en anglais pour obtenir un label prédictif ainsi qu'une probabilité de confiance.
- **Signalement d'erreurs** : Permet d'envoyer les informations de prédiction à **Azure Application Insights** en cas de besoin.
- **Logs centralisés** : Intégration avec **Azure Application Insights** pour collecter et analyser les logs d'erreurs.

## Prérequis

Avant de démarrer, vous devez avoir installé les éléments suivants :

- **Python 3.8 ou supérieur**.
- Le modèle **RoBERTa fine-tuné**.
- Une clé d'instrumentation valide pour **Azure Application Insights**.
- Une **API Backend** accessible à l'URL `http://127.0.0.1:5000/` (ou autre URL configurée).

## Installation

1. Clonez ce dépôt en exécutant la commande suivante :

   git clone https://github.com/Ranack/twit_sentiments
   cd twit_sentiments

2.  Installez les dépendances avec pip :

    
    `pip install -r requirements.txt` 
    
3.  Assurez-vous que l'API Backend fonctionne correctement à l'adresse suivante : `http://127.0.0.1:5000/`.
    

## Configuration

1.  Ajoutez votre clé d'instrumentation **Azure Application Insights** dans le fichier `app.py` :
    

    
    `INSTRUMENTATION_KEY = "votre-clé-d'instrumentation"` 
    
2.  Si nécessaire, mettez à jour l'URL de l'API dans `app.py` :

    
    `API_URL = "http://127.0.0.1:5000/"` 
    

## Utilisation

### En Local

1.  Lancez l'application Streamlit en exécutant la commande suivante :
    

    
    `streamlit run app.py` 
    
2.  Ouvrez votre navigateur et accédez à l'interface utilisateur à l'adresse suivante :
    

    
    `http://localhost:8501` 
    
3.  Saisissez un texte dans la zone prévue et cliquez sur **Prédire** pour obtenir le label et la probabilité associés.
    
4.  Pour signaler une erreur, cliquez sur **Signaler une erreur** après avoir effectué une prédiction.
    

### En Ligne

L'API est également disponible en ligne. Vous pouvez envoyer des requêtes **POST** avec le texte à classifier en utilisant `curl`.

Voici un exemple de commande curl pour interagir avec l'API en ligne :


`curl -X POST https://twisentiment-v2.azurewebsites.net/ \
    -H "Content-Type: application/json" \
    -d '{"text": "I love the new features!"}'` 

-   **URL** : [https://twisentiment-v2.azurewebsites.net/](https://twisentiment-v2.azurewebsites.net/)
-   **Méthode HTTP** : POST
-   **En-têtes** : `Content-Type: application/json`
-   **Données** : JSON avec le texte à classifier, par exemple `{"text": "I love the new features!"}`

#### Exemple de Réponse

La réponse renvoyée par l'API pourrait ressembler à ceci :



`{
  "label": "1",
  "confidence": 0.98
}` 

Cela signifie que le texte a été classé comme positif (1) (0 correspond à un sentiment négatif) avec une probabilité de 98%.

## Structure du Projet

Voici la structure du projet et une brève description des fichiers :



├── .github/
│   └── workflows/
│       └── python-app.yml           # Configuration de GitHub Actions pour CI/CD
├── fine_tuned_roberta/
│   └── pytorch_model.bin            # Modèle RoBERTa fine-tuné (stocké via Git LFS)
├── .dockerignore                    # Liste des fichiers à ignorer lors de la création du Docker image
├── .gitattributes                   # Définition des attributs pour Git LFS
├── .gitignore                       # Liste des fichiers à ignorer par Git
├── API.py                           # Logique du backend pour l'API de classification
├── App.py                           # Interface graphique avec Streamlit (en local)
├── Dockerfile                       # Définition de l'image Docker pour le déploiement
├── requirements.txt                 # Liste des dépendances Python
├── test_api.py                      # Tests unitaires pour l'API backend
└── README.md                        # Documentation du projet` 

## Intégration avec Azure Application Insights

Les logs de signalement d'erreurs sont envoyés à Azure Application Insights. Assurez-vous de vérifier les éléments suivants :

-   La clé d'instrumentation est correcte.
-   Les données envoyées respectent le format JSON attendu par Azure Application Insights.
  


