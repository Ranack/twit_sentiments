import streamlit as st
import requests

# Définir l'URL de votre API FastAPI en local
API_URL = "http://127.0.0.1:5000/predict/"

# Interface Streamlit
st.title("Test de Classification de Texte avec RoBERTa Fine-Tuné")

# Zone de texte pour saisir le texte à prédire
text_input = st.text_area("Entrez le texte à analyser", "")

# Si le bouton "Prédire" est cliqué
if st.button("Prédire"):

    if text_input.strip() != "":
        # Créer la payload pour la requête
        payload = {"text": text_input}

        # Envoyer la requête POST à l'API
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
                st.error("Erreur dans la réponse de l'API")
        except Exception as e:
            st.error(f"Erreur lors de la connexion à l'API : {e}")
    else:
        st.warning("Veuillez entrer un texte à analyser")
