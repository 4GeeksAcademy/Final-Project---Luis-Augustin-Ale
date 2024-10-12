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
    Cargar los modelos.
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
        content = element.get('content',[])                                                                             # para extraer el contenido de 'content'
        item_content = content.get('itemContent',{})                                                                    # para extraer el contenido de 'itemContent'
        if 'tweet_results' not in item_content or 'result' not in item_content['tweet_results'] : continue              # que siga de largo con este elemento si no tiene 'tweet_result' o 'result'
        result = item_content['tweet_results']['result']                                                                # para extraer el contenido de 'tweet_results'
        if 'legacy' not in result or 'full_text' not in result['legacy']: continue                                      # que siga de largo con este elemento si no tiene 'full_text' o 'legacy'
        full_text = result['legacy']['full_text']                                                                       # para extraer el contenido de 'full_text'
        post_date = result['legacy']['created_at']
        likes = result['legacy']['favorite_count']
        clean_data.append((post_date, full_text, likes))
    return clean_data

# to start session_state and state if the search is completed
if 'search_done' not in st.session_state:
    st.session_state.search_done = False

# cover image
st.image(r"C:\Users\Agustín\Desktop\4Geeks\Clases\30. Proyecto Final\Public Environment\Final-Project---Luis-Augustin-Ale\.streamlit\images\portrait.PNG", use_column_width=True)  

# header
st.header("Your Personalized X-Sentiment Analysis")

# to create all tabs once
tab1, tab2, tab3 = st.tabs(["Set-up your Search", "Get Data", "Get Analysis"])

# tab1: search config
with tab1:
    st.subheader("Set up your own personalized search")
    # to show disclaimer
    st.markdown(
        """
        <div style="background-color: rgba(247, 230, 202, 0.8); color: #53463a; padding: 3px; border-radius: 1px; font-size: 11px;">
            ⚠️ This app uses a beta version pretrained model for sentiment analysis and may produce inaccurate results.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write('''
             
             ''')
    keyword = st.text_input("Enter a keyword to search tweets:", "Lebron")
    st.write(' ')
    num_tweets = st.slider("Select the number of tweets to retrieve", 100, 1000, step=50)
    st.write(' ')
    option = st.radio('Tweet options', 
                   ('Latest', 'Top', 'Others'), 
                    index=0, 
                    key='option', horizontal=True)
    st.write(' ')
    
    # to start searching 
    if st.button("Continue"):
        # statement of actions to complete at the momento client click con button 'continue'
        st.session_state.search_done = True  # successful search
        # to pass the keyword to the API as the search phrase
        user_search_phrase = keyword  # User input from the search box
        querystring = {"query": user_search_phrase, "section": 'latest', "limit": "20"}  # Default filters (to be connected later with the st.slider and the st.radio)
        # calling the API
        try:
            response_api_01 = requests.get(url_tweets_search_api_01, headers=headers, params=querystring)
            response_api_01.raise_for_status()
            entries_api_01 = response_api_01.json()['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]['entries']

            # cleanning the API response data
            clean_data = clean_entries_with_dates(entries_api_01)

            # converting the cleaned data into a DataFrame
            df_clean_data = pd.DataFrame(clean_data, columns=['Date', 'Tweet', 'Tweet_Likes'])
        except Exception as e:
                st.error(f"An error occurred: {e}")
                
    # to display results if search was successful
    if st.session_state.search_done:
        st.write(f'Here you have a sample of your "{keyword}" tweets search')
        st.write(df_clean_data.head(5))
        st.write(' ')
        st.write(f"To view the complete results of <{num_tweets}> tweets search based on the option <{option}>, please go to the 'Get Data' tab placed on header.")
        st.write(' ')
        st.write("If you are looking for a comprehensive data analysis of this results, please go to the 'Get Analysis' tab placed on header.")

# tab2: displaying the full dataset and giving the opportunity to download it, not much else
with tab2:
    st.subheader("Data Retrieved")
    if st.session_state.search_done:
        st.write("If you need, here you can download the full data results...")
        st.write(df_clean_data)
        st.write(" ")
        st.write("If you are looking for a comprehensive data analysis of this results, please go to the 'Get Analysis' tab placed on header.")
    else:
        pass

# tab3: Analysing data
with tab3:
    st.subheader("Data Analysis")
    
    if st.session_state.search_done:
        # Performing sentiment analysis on the cleaned data
        # here go results from sentiment analysis, and delete this step code lines
        import numpy as np
        df_clean_data['Sentiment'] = np.random.choice([0, 1], size=len(df_clean_data))
        
        # Next steps, from now on, the code is to keep
        # Displaying sentiment analysis results
        st.write("Sentiment Analysis Results:")
        sentiment_counts = df_clean_data['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)  # Displaying a bar chart of sentiments
        
        # Additional insights
        total_tweets = len(df_clean_data)
        total_likes = df_clean_data['Tweet_Likes'].sum()
        st.write(f"Total Tweets Analyzed: {total_tweets}")
        st.write(f"Sentiment Breakdown: {sentiment_counts.to_dict()}")
        st.write(f"Total Likes on Tweets: {total_likes}")
        
    else:
        st.write('Perform a search in tab "Set-up your Search" to get a personalized data analysis.')
