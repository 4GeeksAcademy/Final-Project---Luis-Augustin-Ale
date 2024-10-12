import pandas as pd
import streamlit as st  # Agregar esta línea
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# cloud charts (positive and negative sentiments)
def plot_wordcloud(df):
    if 'Tweet' not in df.columns or 'Sentiment' not in df.columns:
        st.error("the Dataframe's structure is not correct.")
        return
    
    positive_words = " ".join(df['Tweet'][df['Sentiment'] == 1])
    negative_words = " ".join(df['Tweet'][df['Sentiment'] == 0])

    wordcloud = WordCloud(width=875, height=900, background_color="black", max_words=50, min_font_size=20, random_state=42)\
        .generate(positive_words)
    
    wordcloud2 = WordCloud(width=875, height=900, background_color="black", max_words=50, min_font_size=20, random_state=42)\
        .generate(negative_words)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor=None)
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax2.imshow(wordcloud2, interpolation='bilinear')
    ax1.set_title('Positive Tweets', fontsize=20)
    ax2.set_title('Negative Tweets', fontsize=20)
    ax1.axis("off")
    ax2.axis("off")
    fig.tight_layout()
    
    st.pyplot(fig)  # Esta línea debe quedar al final para mostrar el gráfico en Streamlit
