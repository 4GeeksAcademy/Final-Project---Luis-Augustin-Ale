######pass the translated output to the model #####
import torch
from transformers import RobertaTokenizer
from custom_class_final_model import CustomRobertaModel
#call/load model from hugging face

def load_custom_sentiment_model():
  
    try:
        model_name = "AleOfDurin/final_retrained_model"
        model = CustomRobertaModel.from_pretrained(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model.eval()  # Ensure the model is in evaluation mode
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model:{e}") #using st.error in main script for reusability 

# Load the model and tokenizer
model_custom, tokenizer_custom = load_custom_sentiment_model()


#tokenize translated df and pass translated df to the model

def predict_sentiment(model, tokenizer, sentence):
    """
    Use the custom model to predict sentiment of a given sentence.
    """
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Set model to evaluation mode
    model.eval()
    
    # No gradient calculation for inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs  # For custom model, it's just the logits
    
    # Get the predicted class (0 or 1)
    predictions = torch.argmax(logits, dim=1)
    return predictions.item()

def analyze_sentiments(model, tokenizer, df):
    
    label_mapping = {0: "Negative", 1: "Positive"}  
    sentiments = []

    for tweet in df['Tweet']:
        # Get the predicted sentiment label
        predicted_label = predict_sentiment(model, tokenizer, tweet)
        sentiment = label_mapping.get(predicted_label, "Unknown")
        sentiments.append(sentiment)
    
    # Add the new column to the DataFrame
    df['Sentiment'] = sentiments
    return df