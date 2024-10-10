import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
from tqdm import tqdm  # For progress tracking
import time


def load_data():
    """Load the tokenized datasets."""
    train_encodings, train_labels = torch.load(r'C:/Users/aless/Desktop/final project/Final-Project---Luis-Augustin-Ale/notebooks/train_encodings.pt')
    val_encodings, val_labels = torch.load(r'C:/Users/aless/Desktop/final project/Final-Project---Luis-Augustin-Ale/notebooks/val_encodings.pt')

    return train_encodings, train_labels, val_encodings, val_labels


def prepare_dataloaders(train_encodings, train_labels, val_encodings, val_labels, batch_size, num_workers):
    """Prepare train and validation dataloaders."""
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def create_model():
    """Create and load the RoBERTa model with some layers frozen."""
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Freeze first 8 layers instead of 10 for more flexibility in fine-tuning
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    for i in range(8):  # Unfreezing more layers for better training
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

    # Unfreeze the last 4 layers
    for i in range(8, 12):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = True

    # Add Dropout and Classification Layer
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),  # Dropout to avoid overfitting
        torch.nn.Linear(model.config.hidden_size, 2)  # Binary classification layer (2 labels: positive/negative)
    )

    return model


def setup_optimizer_scheduler(model, train_loader, epochs):
    """Set up the optimizer and scheduler."""
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Total number of training steps = number of batches * number of epochs
    num_training_steps = len(train_loader) * epochs

    # Scheduler: linear learning rate decay
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    return optimizer, scheduler


def train_model(model, train_loader, optimizer, scheduler, device, epochs, gradient_accumulation_steps, scaler):
    """Main training loop for the model."""
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        # Progress bar for training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            labels = labels.long()  # Convert labels to LongTensor


            # Enable mixed precision for faster computation
            with autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract the logits corresponding to the [CLS] token (first token)
                cls_logits = outputs.logits[:, 0, :]  # Take only the logits for the [CLS] token
                loss = torch.nn.functional.cross_entropy(cls_logits, labels)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)  # Apply scaling
                scaler.update()  # Update scaling factors
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. Average loss: {avg_loss:.4f}, Time elapsed: {time.time() - epoch_start_time:.2f} seconds")




def main():
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/fine_tuning_roberta')

    # Set batch size and number of workers
    batch_size = 256
    num_workers = 4

    # Load data and prepare DataLoader
    train_encodings, train_labels, val_encodings, val_labels = load_data()
    train_loader, val_loader = prepare_dataloaders(train_encodings, train_labels, val_encodings, val_labels, batch_size, num_workers)

    # Create and move model to device
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize optimizer, scheduler, and mixed precision components
    optimizer, scheduler = setup_optimizer_scheduler(model, train_loader, epochs=3)
    scaler = GradScaler()

    # Train the model
    train_model(model, train_loader, optimizer, scheduler, device, epochs=3, gradient_accumulation_steps=2, scaler=scaler)

    # Save the model
    torch.save(model.state_dict(), '2nd_model.pth')
    print("Model saved successfully.")

    # Close TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
