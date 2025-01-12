import os
import streamlit as st
import requests
from datetime import datetime

# Fonction pour envoyer les logs à Azure Application Insights
def send_log_to_application_insights(prediction_info):
    payload = {
        "iKey": INSTRUMENTATION_KEY,
        "name": "Microsoft.ApplicationInsights.Message",
        "time": prediction_info['timestamp'],
        "data": {
            "baseType": "MessageData",
            "baseData": {
                "message": f"Prédiction: {prediction_info['text']} - Label prédit: {prediction_info['predicted_label']} - Confiance: {prediction_info['confidence']}",
                "severityLevel": 3  # Niveau de sévérité (3 = Avertissement, 4 = Erreur, etc.)
            }
        }
    }

    # Envoi de la requête POST à Application Insights
    try:
        response = requests.post(APPLICATION_INSIGHTS_URL, json=payload)
        
        # Retourner la réponse pour l'afficher dans Streamlit
        return response  # On retourne la réponse pour pouvoir l'afficher dans l'interface Streamlit
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'envoi des logs : {e}")
        return None  # Si l'envoi échoue, on retourne None

# Clé d'instrumentation d'Application Insights pour usage local
INSTRUMENTATION_KEY = "637d1141-3dbe-4693-8989-df89eb0c6520"  # Remplacez par votre clé d'instrumentation Azure

# Vérifiez si la clé existe
if not INSTRUMENTATION_KEY:
    st.error("La clé d'instrumentation Azure est manquante.")
    st.stop()

# Définir l'URL de votre API FastAPI sur Azure ou en local (avec le port 5000 en local)
API_URL = "http://127.0.0.1:5000/"  # Utilisation de l'URL racine

# URL d'Azure Application Insights (remplacez par la vôtre)
APPLICATION_INSIGHTS_URL = "https://canadacentral-1.in.applicationinsights.azure.com/v2/track"  # URL de tracking de l'Application Insights

# Interface Streamlit
st.title("Test de Classification de Texte avec RoBERTa Fine-Tuné")

# Zone de texte pour saisir le texte à prédire avec un exemple de texte par défaut
text_input = st.text_area("Entrez le texte à analyser (en anglais)", "Exemple : I love this product!")

# Si le bouton "Prédire" est cliqué
if st.button("Prédire"):
    if text_input.strip() != "":
        # Créer la payload pour la requête
        payload = {"text": text_input}

        # Afficher un message de chargement
        with st.spinner("Envoi de la requête à l'API..."):
            try:
                # Augmenter le timeout pour éviter les erreurs de connexion
                response = requests.post(API_URL, json=payload, timeout=10)

                # Vérifier la réponse de l'API
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Résultats de la prédiction")
                    st.write(f"Texte : {result['text']}")
                    st.write(f"Label prédit : {result['predicted_label']}")
                    st.write(f"Confiance : {result['confidence']:.2f}")

                    # Sauvegarder les résultats dans `st.session_state` pour garder les informations
                    st.session_state.prediction_info = {
                        "text": result["text"],
                        "predicted_label": result["predicted_label"],
                        "confidence": result["confidence"]
                    }
                else:
                    st.error(f"Erreur dans la réponse de l'API: {response.status_code}")
                    st.write(response.text)  # Affiche la réponse complète pour débogage
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de la connexion à l'API : {e}")
                st.write(f"Exception: {e}")  # Affiche l'exception pour mieux comprendre l'erreur
    else:
        st.warning("Veuillez entrer un texte à analyser")

# Vérifier si les données de prédiction existent et afficher l'option pour signaler une erreur
if 'prediction_info' in st.session_state:
    if st.button("Signaler une erreur"):
        # Préparer les informations à envoyer à Azure Application Insights
        prediction_info = st.session_state.prediction_info
        prediction_info['timestamp'] = datetime.utcnow().isoformat()

        # Appeler la fonction d'envoi des logs
        response_log = send_log_to_application_insights(prediction_info)

        # Vérification de la réponse de l'envoi
        if response_log:
            st.success("Erreur de prédiction remontée, merci de votre coopération ! :)")
        else:
            st.error("Il y a eu un problème lors de l'envoi des logs.")
