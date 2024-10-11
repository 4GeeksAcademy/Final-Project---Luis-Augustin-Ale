import os
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaModel
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

# Custom Roberta Model class

class CustomRobertaModel(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomRobertaModel, self).__init__()
        # Load the pre-trained Roberta base model
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
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return logits

def freeze_unfreeze_layers(model):
    # Freeze first 8 layers of RoBERTa
    for param in model.roberta.roberta.embeddings.parameters():
        param.requires_grad = False

    for i in range(8):
        for param in model.roberta.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

    # Unfreeze last 4 layers
    for i in range(8, 12):
        for param in model.roberta.roberta.encoder.layer[i].parameters():
            param.requires_grad = True

# Function to load the data
def load_data():
    """Load the tokenized datasets."""
    train_encodings, train_labels = torch.load(r'C:\Users\aless\Desktop\final project 2.1\train_encodings.pt')
    val_encodings, val_labels = torch.load(r'C:\Users\aless\Desktop\final project 2.1\val_encodings.pt')

    return train_encodings, train_labels, val_encodings, val_labels

# Function to prepare dataloaders
def prepare_dataloaders(train_encodings, train_labels, val_encodings, val_labels, batch_size, num_workers):
    """Prepare train and validation dataloaders."""
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

# Function to set up optimizer and scheduler
def setup_optimizer_scheduler(model, train_loader, epochs):
    """Set up the optimizer and scheduler."""
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Total number of training steps = number of batches * number of epochs
    num_training_steps = len(train_loader) * epochs

    # Scheduler: linear learning rate decay
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    return optimizer, scheduler

# Modified training loop with loss and learning rate tracking
def train_model(model, train_loader, optimizer, scheduler, device, epochs, gradient_accumulation_steps, scaler):
    """Main training loop for the model with logging."""
    log_file_path = "training_log.txt"  # Log file for tracking loss and learning rate

    with open(log_file_path, "w") as log_file:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            epoch_start_time = time.time()

            # Progress bar for training loop
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for step, batch in enumerate(progress_bar):
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                labels = labels.long()  # Convert labels to LongTensor

                # Enable mixed precision for faster computation
                with autocast(dtype=torch.float16):
                    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Backward pass with scaled gradients
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)  # Apply scaling
                    scaler.update()  # Update scaling factors
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()

                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]

                # Log loss and learning rate after each batch
                log_message = f"Epoch {epoch + 1}, Step {step + 1}, Loss: {avg_loss:.4f}, LR: {current_lr:.8f}"
                log_file.write(log_message + "\n")

                # Update progress bar
                progress_bar.set_postfix({'loss': avg_loss, 'lr': current_lr})

            avg_epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}, Time: {time.time() - epoch_start_time:.2f} seconds")

    print(f"Training complete. Loss and learning rate logged to {log_file_path}")

# Function to save the custom model
def save_custom_model(model, save_directory):
    """Save the entire custom model and its config."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the RoBERTa model part
    model.roberta.save_pretrained(save_directory)

    # Save the additional layers manually
    torch.save(model.additional_layer_1.state_dict(), os.path.join(save_directory, 'additional_layer_1.pth'))
    torch.save(model.additional_layer_2.state_dict(), os.path.join(save_directory, 'additional_layer_2.pth'))
    torch.save(model.classifier.state_dict(), os.path.join(save_directory, 'classifier.pth'))

    # Save the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained(save_directory)

    print(f"Model, additional layers, and tokenizer saved at {save_directory}")

# Main function to train and save the model
def main():
    # Load data
    train_encodings, train_labels, val_encodings, val_labels = load_data()

    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(train_encodings, train_labels, val_encodings, val_labels, batch_size=256, num_workers=4)

    # Initialize custom model
    model_custom = CustomRobertaModel(num_labels=2)

    # Move model to device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_custom.to(device)

    # Freeze/Unfreeze layers
    freeze_unfreeze_layers(model_custom)

    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer_scheduler(model_custom, train_loader, epochs=3)
    scaler = GradScaler()

    # Train the model
    train_model(model_custom, train_loader, optimizer, scheduler, device, epochs=3, gradient_accumulation_steps=2, scaler=scaler)

    # Save the custom model after training
    save_custom_model(model_custom, r'C:\Users\aless\Desktop\final project 2.1\custom_roberta_retrained')

if __name__ == '__main__':
    main()
