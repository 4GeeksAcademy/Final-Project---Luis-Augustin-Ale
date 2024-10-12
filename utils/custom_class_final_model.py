# Note: This file only contains the class definition. To use it, you will import it in another script.

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, PreTrainedModel, RobertaModel,RobertaForSequenceClassification
from safetensors.torch import load_file
from huggingface_hub import PyTorchModelHubMixin

class CustomRobertaModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_labels=2):
        super(CustomRobertaModel, self).__init__()
        # Load the pre-trained RobertaForSequenceClassification model
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        
        # Additional dense layers
        self.additional_layer_1 = nn.Linear(self.roberta.config.hidden_size, 512)
        self.additional_layer_2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_labels)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through Roberta model (get the hidden states)
        outputs = self.roberta.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # Extract the [CLS] token's output (pooled output) from the last hidden state
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the hidden state for the [CLS] token
        
        # Pass through the additional layers
        x = self.relu(self.additional_layer_1(pooled_output))
        x = self.relu(self.additional_layer_2(x))
        
        # Final classification layer
        logits = self.classifier(x)

        return logits

