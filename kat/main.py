import torch
import json
import katransformer
import pandas as pd
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder
from katransformer import create_kat_amp_model

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained KAT model for AMP prediction
model = create_kat_amp_model(seq_length=100, embed_dim=768, num_classes=1)
model.load_state_dict(torch.load("amp_prediction_model.pth"))
model = model.to(device)
model.eval()


# Helper function to encode sequences
def encode_sequence(sequence):
    return torch.tensor(
        [ord(char) - ord("A") + 1 for char in sequence], dtype=torch.long
    )


# Define the sequences to test (for demonstration)
amp_sequences = [
    "SLFSLIKAGAKFLGKNLLKQGACYAACKASKQC",
    "GIMDTVKNVAKNLAGQLLDKLKCKITAC",
    "GYGCPFNQYQCHSHCSGIRGYKGGYCKGTFKQTCKCY",
]

# Process each sequence and make predictions
for seq in amp_sequences:
    # Encode sequence and move to device
    encoded_seq = encode_sequence(seq).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(encoded_seq)
        probability = torch.sigmoid(output).item()  # Convert logits to probability

    # Print the sequence and the prediction probability
    print(f"Sequence: {seq}")
    print(f"AMP Prediction Probability: {probability:.4f}")
