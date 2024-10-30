import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from katransformer import create_kat_amp_model  # Import the custom KAT model
from sklearn.model_selection import train_test_split


# Custom Dataset Class for AMP Sequences
# This class loads amino acid sequences from a CSV file and encodes them for model input.
class AMPDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Load data from CSV
        self.transform = transform

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


# Function to find the maximum sequence length in the dataset
# This function scans the dataset to determine the longest sequence.
def find_max_seq_length(csv_file):
    data = pd.read_csv(csv_file)
    max_length = data["aa_seq"].apply(len).max()
    return max_length


# Collate function to handle variable sequence lengths by padding
# Pads each batch to the length of the longest sequence within that batch.
def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_length = max(len(seq) for seq in sequences)  # Max length in the batch
    padded_seqs = torch.zeros(
        len(sequences), max_length, dtype=torch.long
    )  # Pad with zeros
    for i, seq in enumerate(sequences):
        end = len(seq)
        padded_seqs[i, :end] = seq  # Pad each sequence to max length in the batch
    return padded_seqs, torch.tensor(labels)


# Calculate maximum sequence length across both train and validation datasets
# Ensures the model's `seq_length` parameter is set to the longest sequence in either dataset.
csv_file_path = "data.csv"  # Replace with actual path
train_data, val_data = train_test_split(pd.read_csv(csv_file_path), test_size=0.2)

train_max_length = (
    train_data["aa_seq"].apply(len).max()
)  # Max sequence length in training set
val_max_length = (
    val_data["aa_seq"].apply(len).max()
)  # Max sequence length in validation set
max_seq_length = max(
    train_max_length, val_max_length
)  # Longest length across both sets
print(f"Maximum sequence length for training and validation: {max_seq_length}")

# Save the max_seq_length to a file for validation use
# This ensures consistency when loading the model for validation or testing.
with open("max_seq_length.txt", "w") as f:
    f.write(str(max_seq_length))

# Save split data to CSV files
train_data.to_csv("train_amp.csv", index=False)
val_data.to_csv("val_amp.csv", index=False)

# Prepare datasets and data loaders
train_dataset = AMPDataset("train_amp.csv")
val_dataset = AMPDataset("val_amp.csv")

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

# Define model and move to device
# Initialize the custom KAT model, setting the maximum sequence length.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_kat_amp_model(
    seq_length=max_seq_length, embed_dim=768, num_classes=1
)  # Set max sequence length
model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # Step learning rate scheduler

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

        # Forward pass
        outputs = model(inputs)  # Get model outputs
        loss = criterion(outputs, labels)  # Compute binary cross-entropy loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss for reporting

    scheduler.step()  # Adjust learning rate

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(
        f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}"
    )

# Save the trained model
# The model's state dictionary is saved to a file for later evaluation or deployment.
torch.save(model.state_dict(), "amp_prediction_model.pth")
