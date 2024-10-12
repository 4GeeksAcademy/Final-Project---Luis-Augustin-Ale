import pandas as pd
import seaborn as sns
import nltk
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import toml

secrets = toml.load(r'C:\Users\aless\Desktop\final project\Final-Project---Luis-Augustin-Ale\.streamlit\secrets.toml')

df = pd.read_csv("hf://datasets/PrkhrAwsti/Twitter_Sentiment_3M/twitter_dataset.csv")

download("wordnet")
lemmatizer = WordNetLemmatizer()
download("stopwords")
stop_words = set(stopwords.words("english"))

def limpiar_tweet(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Eliminar menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # Eliminar caracteres que no sean letras, números, espacios, o hashtags
    text = re.sub(r'[^a-z0-9# ]', ' ', text)
    
    # Eliminar letras sueltas (que suelen ser ruido)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    
    # Reducir espacios múltiples a uno
    text = re.sub(r'\s+', ' ', text).strip()

    return text.split()

def lemmatizar_text(tweets, lemmatizer=lemmatizer, stop_words=stop_words):
    
    # Lematizar los tweets
    tokens = [lemmatizer.lemmatize(tweet) for tweet in tweets]
    
    # Eliminar stopwords
    tokens = [tweet for tweet in tokens if tweet not in stop_words]
    
    # Eliminar tweets con longitud menor o igual a 3 caracteres
    tokens = [tweet for tweet in tokens if len(tweet) > 3]
    
    return tokens

df=df.drop(columns=["Unnamed: 0"],axis=1)

df=df.dropna()
df=df.drop_duplicates()   #añadido despues by Ale (en caso que se rompa algo)

df['sentiment'] = df['sentiment'].astype(int)

df=df[df["tweet"]!=""]

df=df[df["sentiment"]!=2]

df['proTweets']=df['tweet'].apply(limpiar_tweet)

df=df.drop(columns=["tweet"],axis=1)

df['lemantizedTweets']=df['proTweets'].apply(lemmatizar_text)

df=df.drop(columns=["proTweets"],axis=1)

lista_tweets=df['lemantizedTweets']                                           
lista_tweets = [" ".join(tweet) for tweet in lista_tweets]
X_train, X_test, y_train, y_test = train_test_split(lista_tweets, df["sentiment"], test_size = 0.2, random_state = 42)
vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.8, min_df = 5)
X_train_vectorized = vectorizer.fit_transform(X_train)

""" lista_tweets=df['lemantizedTweets']
lista_tweets = [" ".join(tweet) for tweet in lista_tweets]
vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.8, min_df = 5)
X = vectorizer.fit_transform(lista_tweets)
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) """

#modificado bajo consejo de felix 


print("El script se ha ejecutado correctamente.")
print(X_train_vectorized.shape)

