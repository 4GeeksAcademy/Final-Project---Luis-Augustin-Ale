from deep_translator import GoogleTranslator
translator = GoogleTranslator()
#df_prueba_roberta.rename(columns={'sentiment': 'tweets'}, inplace=True)

def traducir(text):
    # Detectar el idioma automáticamente y traducirlo al inglés
    if GoogleTranslator(source='auto', target='en').translate(text):
        return GoogleTranslator(source='auto', target='en').translate(text)
    else:
        return text

df_prueba_roberta['tweets_traducidos'] = df_prueba_roberta['tweets'].apply(traducir)
