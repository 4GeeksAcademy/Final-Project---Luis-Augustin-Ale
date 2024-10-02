import streamlit as st
import requests
import pandas as pd

# API configuration
url_tweets_search_api_01 = "https://twitter-x.p.rapidapi.com/search/"
headers = {
    "x-rapidapi-key": "2e2c904e0cmshdbad92d97808688p1e798ajsna8b04a4dd68a", 
    "x-rapidapi-host": "twitter-x.p.rapidapi.com"
}

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

def main():
    st.title("Tweets Sentiment Test")

    # Search box for keyword 
    keyword = st.text_input("Enter a keyword to search tweets:")

    # Slider to select the amount of tweets
    num_tweets = st.slider("Select the number of tweets to retrieve", 100, 1000, step=50)

    
    option = st.radio('Tweet options', 
                        ('Latest', 'Top', 'Others'), 
                        index=0, 
                        key='option', horizontal=True)

    # Disclaimer text
    st.caption("This app uses a pretrained model for sentiment analysis and may produce inaccurate results.")

    # Search button
    if st.button("Search"):
        # Check if the user has entered a keyword
        if not keyword:
            st.warning("Please enter a keyword to search for tweets.")
        else:
            # Pass the keyword to the API as the search phrase
            user_search_phrase = keyword  # User input from the search box
            querystring = {"query": user_search_phrase, "section": 'latest', "limit": "20"}  # Default filters

            # Call the API
            try:
                response_api_01 = requests.get(url_tweets_search_api_01, headers=headers, params=querystring)
                response_api_01.raise_for_status()
                entries_api_01 = response_api_01.json()['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]['entries']

                # Clean the API response data
                clean_data = clean_entries_with_dates(entries_api_01)

                # Convert the cleaned data into a DataFrame and display it in Streamlit
                df_clean_data = pd.DataFrame(clean_data, columns=['Date', 'Tweet'])
                st.write(df_clean_data)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
