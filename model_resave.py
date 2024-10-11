# Step 1: Import necessary libraries
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Step 2: Load the trained model and the tokenizer

# Path to the model weights and the path to save the full model
model_weights_path = r'C:\Users\aless\Desktop\final project 2.1\sentiment_fine_tuned.pth'
save_path = r'C:\Users\aless\Desktop\final project 2.1\full_model'

# Load the tokenizer (Roberta's tokenizer)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Initialize a Roberta model with sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Modify the classifier to match the fine-tuned model's structure
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.config.hidden_size, 2)  # Fine-tuned classifier layer
)

# Load the fine-tuned weights into the modified model
model.load_state_dict(torch.load(model_weights_path))

# Step 3: Save the full model, configuration, and tokenizer

# Save the full model (weights, config, etc.) and tokenizer for Hugging Face
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Full model and tokenizer saved at {save_path}.")
