###Copy the code for safety, to avoid braking your work :D ###

import streamlit as st
import requests
import pandas as pd
from translate_API_output import traducir, traducir_tweets
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from deep_translator import GoogleTranslator

import streamlit as st
import torch
from transformers import RobertaTokenizer
from custom_class_final_model import CustomRobertaModel
from model_load_apply import load_custom_sentiment_model, predict_sentiment, analyze_sentiments


# API configuration
url_tweets_search_api_01 = "https://twitter-x.p.rapidapi.com/search/"
headers = {
    "x-rapidapi-key": "2e2c904e0cmshdbad92d97808688p1e798ajsna8b04a4dd68a", 
    "x-rapidapi-host": "twitter-x.p.rapidapi.com"
}

### MODEL  FROM HUGGINGfaces ###
@st.cache_resource
def get_model_and_tokenizer():
    try:
        model_custom, tokenizer_custom = load_custom_sentiment_model()
    except RuntimeError as e:
        st.error(str(e))
        return None, None
    
    return model_custom, tokenizer_custom
# Call the cached function
model_custom, tokenizer_custom = get_model_and_tokenizer()
# Check if the model was loaded successfully, otherwise exit
if model_custom is None or tokenizer_custom is None:
        st.stop()  # Stop the app if the model couldn't be loaded



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
if 'df_clean_data' not in st.session_state:
    st.session_state.df_clean_data = None

# cover image
st.image(r"..\.streamlit\images\cover logo text.png", use_column_width=True)  

# header
st.markdown("<p style='text-align: center; font-size:22px; font-weight: bold;'>Tailored Sentiment Analysis at Your Fingertips</p>", unsafe_allow_html=True)

# to create all tabs once
tab1, tab2, tab3 = st.tabs(["Set-up your Search", "Get Data", "Get Analysis"])

# tab1: search config
with tab1:
    st.markdown("<p style='text-align: center; font-size:18px; font-weight: bold;'>Set up your search</p>", unsafe_allow_html=True)
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

    # continue_disabled = not keyword.strip() a ver si la quito
    
    # to start searching 
    if st.button("Continue"):
        if not keyword.strip():
            st.warning("You can't search with an empty keyword. Please enter a keyword")
        else:
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
                df_clean_data['Tweet']=df_clean_data['Tweet'].apply(traducir)
                st.session_state.df_clean_data=df_clean_data

            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    # to display results if search was successful
    if st.session_state.search_done:
        df_clean_data=st.session_state.df_clean_data
        # Check dataframe not none and empty
        if df_clean_data is not None and not df_clean_data.empty:
            if keyword.strip():
                st.write(f'Here you have a sample of your "{keyword}" tweets search')
                st.write(df_clean_data.head(5))
                st.write(' ')
                st.write(f"To view the complete results of <{num_tweets}> tweets search based on the option <{option}>, please go to the 'Get Data' tab placed on header.")
                st.write(' ')
                st.write("If you are looking for a comprehensive data analysis of this results, please go to the 'Get Analysis' tab placed on header.")
        else:
            st.warning("No tweets were found for the current search.")
        

# LUIS -----------------------------------------------------------------------------------------------------------------------------------------------
# Recibis un dataset con df_clean_data[index, 'Date', 'Tweet', 'Tweet_Likes'] y das como output df_clean_data[index, 'Date', 'Tweet', 'Tweet_Likes'] 
# con la columna 'Tweet' ya con todo traducido
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# df_clean_data['Translated_Tweet']=df_clean_data['Tweet'].apply(traducir) a borrar si funciona


# tab2: displaying the full dataset and giving the opportunity to download it, not much else
with tab2:
    st.subheader("Data Retrieved")
    if st.session_state.search_done:
        df_clean_data=st.session_state.df_clean_data
        if df_clean_data is not None and not df_clean_data.empty:
            st.write("If you need, here you can download the full data results...")
            st.write(df_clean_data)
            st.write(" ")
            st.write("If you are looking for a comprehensive data analysis of this results, please go to the 'Get Analysis' tab placed on header.")
    else:
        st.warning("No data available to display")
        pass


# ALE ------------------------------------------------------------------------------------------------------------------------------------------------

# Applying the model in  Streamlit
if st.session_state.search_done and model_custom is not None:
    df_clean_data = st.session_state.df_clean_data

    # Ensure DataFrame exists and has content before analysis
    if df_clean_data is not None and not df_clean_data.empty:
        # Analyze sentiments using the loaded model
        df_clean_data = analyze_sentiments(model_custom, tokenizer_custom, df_clean_data)
        
        # Update the session state with the new DataFrame
        st.session_state.df_clean_data = df_clean_data
        
        # Refresh display after adding sentiment analysis
        st.write("Sentiment Analysis Completed:")
        st.write(st.session_state.df_clean_data.head())


# ----------------------------------------------------------------------------------------------------------------------------------------------------


# tab3: Analysing data
with tab3:
    st.subheader("Data Analysis")
    
    if st.session_state.search_done:
        # Performing sentiment analysis on the cleaned data
        # here go results from sentiment analysis, and delete this step code lines
        import numpy as np
        df_clean_data['Sentiment'] = np.random.choice([0, 1], size=len(df_clean_data))
        
        # AGUS ---------------------------------------------------------------------------------------------------------------------------------------------
        
        # Next steps, from now on, the code is to keep
        # Displaying sentiment analysis results
        st.write("Sentiment Analysis Results:")
        sentiment_counts = df_clean_data['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)  # Displaying a bar chart of sentiments
        
        # wordcloud charts (positive and negative sentiments).
        import sys
        import os
        # Add the parent directory to sys.path (one level up)
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agus_temporal')))
        # Now you can import from dashboard_charts.py
        from dashboard_charts import plot_wordcloud
        st.write(df_clean_data.head())
        plot_wordcloud(df_clean_data)
        
        # Additional insights
        total_tweets = len(df_clean_data)
        total_likes = df_clean_data['Tweet_Likes'].sum()
        st.write(f"Total Tweets Analyzed: {total_tweets}")
        st.write(f"Sentiment Breakdown: {sentiment_counts.to_dict()}")
        st.write(f"Total Likes on Tweets: {total_likes}")
                
    else:
        st.warning('Perform a search in tab "Set-up your Search" to get a personalized data analysis.')
