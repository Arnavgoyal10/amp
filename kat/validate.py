import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from katransformer import create_kat_amp_model  # Import the custom KAT model


# Custom Dataset Class for AMP Sequences
# This class loads amino acid sequences from a CSV file, encodes them, and prepares labels.
class AMPDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)  # Load data from CSV file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["aa_seq"]  # Get amino acid sequence
        label = self.data.iloc[idx]["AMP"]  # Get label indicating AMP or not

        # Encode sequence as integer values (A=1, B=2, ..., Z=26)
        encoded_seq = torch.tensor(
            [ord(char) - ord("A") + 1 for char in sequence], dtype=torch.long
        )
        label = torch.tensor(
            int(label == "True"), dtype=torch.float
        )  # Convert "True"/"False" label to binary (1 or 0)

        return encoded_seq, label


# Collate function to handle variable sequence lengths by padding
# Pads each batch to the length of the longest sequence within that batch.
def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_length = max(len(seq) for seq in sequences)  # Find max length in the batch
    padded_seqs = torch.zeros(
        len(sequences), max_length, dtype=torch.long
    )  # Pad with zeros
    for i, seq in enumerate(sequences):
        end = len(seq)
        padded_seqs[i, :end] = seq  # Pad each sequence to max length in the batch
    return padded_seqs, torch.tensor(labels)


# Load maximum sequence length from file to ensure consistency with training
# This file ensures that validation uses the same sequence length as training.
with open("max_seq_length.txt", "r") as f:
    max_seq_length = int(f.read().strip())
print(f"Maximum sequence length for validation: {max_seq_length}")

# Load validation dataset and prepare DataLoader
val_dataset = AMPDataset("val_amp.csv")
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

# Load model with matched sequence length
# Initialize the model with the sequence length used during training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_kat_amp_model(seq_length=max_seq_length, embed_dim=768, num_classes=1)
model.load_state_dict(
    torch.load("amp_prediction_model.pth", map_location=device)
)  # Load trained weights
model.to(device)
model.eval()  # Set model to evaluation mode

# Define loss function and metric tracking
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for validation
all_labels = []
all_preds = []

# Validation loop
val_loss = 0.0
with torch.no_grad():  # No gradients needed for validation
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass through the model
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # Compute binary cross-entropy loss
        val_loss += loss.item()  # Accumulate validation loss

        # Apply sigmoid to outputs to get probabilities, then threshold at 0.5
        preds = torch.sigmoid(outputs).cpu().numpy() > 0.1  # Threshold predictions
        all_labels.extend(labels.cpu().numpy())  # Store true labels
        all_preds.extend(preds)  # Store predicted labels

# Calculate evaluation metrics
# Using sklearn to compute accuracy, precision, recall, and F1 score based on predictions.
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

# Print validation loss and performance metrics
print(f"Validation Loss: {val_loss/len(val_loader)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
