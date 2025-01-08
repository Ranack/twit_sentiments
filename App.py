import streamlit as st
import requests

# Définir l'URL de votre API FastAPI sur Azure (avec le port 80)
API_URL = "http://twisentiment-v2.azurewebsites.net:80/predict/"

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
                response = requests.post(API_URL, json=payload)

                # Vérifier la réponse de l'API
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Résultats de la prédiction")
                    st.write(f"Texte : {result['text']}")
                    st.write(f"Label prédit : {result['predicted_label']}")
                    st.write(f"Confiance : {result['confidence']:.2f}")
                else:
                    st.error(f"Erreur dans la réponse de l'API: {response.status_code}")
                    st.write(response.text)  # Affiche la réponse complète pour débogage
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de la connexion à l'API : {e}")
                st.write(f"Exception: {e}")  # Affiche l'exception pour mieux comprendre l'erreur
    else:
        st.warning("Veuillez entrer un texte à analyser")
