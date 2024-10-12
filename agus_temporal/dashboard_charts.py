import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# cloud charts (positive and negative sentiments)
def plot_wordcloud(df):
    positive_words = df['Tweet'][df['Sentiment']==1]
    negative_words = df['Tweet'][df['Sentiment']==0]
    wordcloud = WordCloud(width = 875, height = 900, background_color = "black", max_words = 50, min_font_size = 20, random_state = 42)\
        .generate(str(positive_words))
    wordcloud2 = WordCloud(width = 875, height = 900, background_color = "black", max_words = 50, min_font_size = 20, random_state = 42)\
        .generate(str(negative_words))

    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18, 9), facecolor=None)
    ax1.imshow(wordcloud)
    ax2.imshow(wordcloud2)
    ax1.set_title('positive_words', fontsize=20)
    ax2.set_title('negative_words', fontsize=20)
    ax1.axis("off")
    ax2.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    
    


