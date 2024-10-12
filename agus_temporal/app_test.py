import streamlit as st
import requests
import pandas as pd

# API configuration
url_tweets_search_api_01 = "https://twitter-x.p.rapidapi.com/search/"
headers = {
    "x-rapidapi-key": "2e2c904e0cmshdbad92d97808688p1e798ajsna8b04a4dd68a", 
    "x-rapidapi-host": "twitter-x.p.rapidapi.com"
}


''' ESTA PARTE IMPORTA LOS MODELOS QUE HAYA QUE IMPORTAR, PERSONALIZAR UNA VEZ OBTENIDO NUESTRO ROBERTA
@st.cache_resource
def load_models():
    """
    Cargar los modelos .
    """
    return 'Respuesta modelo de texto', 'Respuesta modelo multimodal'

def get_text_response(model=None, prompt=None, config=None, stream=True):
    """
    Obtener respuesta de texto 
    """
    return "Respuesta de texto del modelo de texto."

def get_vision_response(model=None, prompt_list=None, config={}, stream=True):
    """
    Obtener respuesta 
    """
    return "Respuesta de imagen del modelo de multimodal."
'''


# Function to clean the entries and extract date and text
def clean_entries_with_dates(list_of_elem):
    clean_data = []                                                                              
    for element in list_of_elem:                                                                  
        content = element.get('content', [])                                                      
        item_content = content.get('itemContent', {})                                             
        if 'tweet_results' not in item_content or 'result' not in item_content['tweet_results']: 
            continue                                                                              
        result = item_content['tweet_results']['result']                                          
        if 'legacy' not in result or 'full_text' not in result['legacy']: 
            continue                                                                              
        full_text = result['legacy']['full_text']                                                 
        post_date = result['legacy']['created_at']                                                
        clean_data.append((post_date, full_text))                                                  
    return clean_data


# Inicializar session_state para manejar el cambio de pestañas
if 'page' not in st.session_state:
    st.session_state.page = 'setup'

# Titulo e Inicializar modelos
st.header("Your Personalized X-Sentiment Analysis")

# Lógica para decidir qué pestaña mostrar
if st.session_state.page == 'setup':
    tab1, _ , _ = st.tabs(["Set-up your Search", "Get Data", "Get Analysis"])
elif st.session_state.page == 'data':
    _, tab2, _ = st.tabs(["Set-up your Search", "Get Data", "Get Analysis"])
else:
    _, _, tab3 = st.tabs(["Set-up your Search", "Get Data", "Get Analysis"])

# Pestaña 1: Configuración de búsqueda
if st.session_state.page == 'setup':
    with tab1:
        st.subheader("Set up your own personalized search")
        st.write("Disclaimer: This app uses a pretrained model for sentiment analysis and may produce inaccurate results.")
        keyword = st.text_input("Enter a keyword to search tweets:", "Adidas")
        num_tweets = st.slider("Select the number of tweets to retrieve", 100, 1000, step=50)
        option = st.radio('Tweet options', 
                       ('Latest', 'Top', 'Others'), 
                        index=0, 
                        key='option', horizontal=True)
        
        if st.button("Continue"):
            # Cambiamos el estado de la aplicación a 'data'
            st.session_state.page = 'data'

# Pestaña 2: Mostrar Datos
if st.session_state.page == 'data':
    with tab2:
        st.subheader("Data Retrieved")
        st.write("Aquí se mostrarán los datos obtenidos.")

# Pestaña 3: Análisis de datos
if st.session_state.page == 'analysis':
    with tab3:
        st.subheader("Data Analysis")
        st.write("Aquí va el análisis de sentimientos basado en los tweets.")