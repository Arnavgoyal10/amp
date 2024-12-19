import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_
import pandas as pd  # Add this import for CSV functionality


# Custom Rational Activation Function (as a substitute for KAN layer rational functions)
# The KAT paper suggests using rational functions instead of standard activation functions
# to enhance expressiveness and efficiency. Here, we create a rational activation function
# by parameterizing the numerator and denominator, allowing the function to approximate complex shapes.
class RationalActivation(nn.Module):
    def __init__(self):
        super(RationalActivation, self).__init__()
        # Initialize numerator and denominator coefficients to approximate a ReLU-like function
        self.p = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0])
        )  # Numerator coefficients
        self.q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))  # Denominator coefficients

    def forward(self, x):
        # Define the rational function for activation: numerator / denominator
        numerator = self.p[0] + self.p[1] * x + self.p[2] * x**2 + self.p[3] * x**3
        denominator = 1 + torch.abs(self.q[0] * x + self.q[1] * x**2 + self.q[2] * x**3)
        return numerator / denominator


# Group KAN Layer with Rational Activations
# The KAT model uses Kolmogorov–Arnold Network (KAN) layers to replace standard MLPs in the transformer blocks.
# We implement grouped KANs, where each group has separate linear transformations followed by rational activations.
class GroupKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_groups=8):
        super(GroupKANLayer, self).__init__()
        self.num_groups = num_groups
        # Divide input and output features across groups for efficient parameterization
        group_in_features = in_features // num_groups
        group_out_features = out_features // num_groups

        # Initialize linear layers and rational activations for each group
        self.linears = nn.ModuleList(
            [
                nn.Linear(group_in_features, group_out_features)
                for _ in range(num_groups)
            ]
        )
        self.activations = nn.ModuleList(
            [RationalActivation() for _ in range(num_groups)]
        )

    def forward(self, x):
        # Split input features into groups for independent transformation
        group_inputs = torch.chunk(x, self.num_groups, dim=-1)
        # Apply each group's linear transformation and rational activation, then concatenate results
        group_outputs = [
            activation(linear(g_in))
            for g_in, linear, activation in zip(
                group_inputs, self.linears, self.activations
            )
        ]
        return torch.cat(group_outputs, dim=-1)


# KAT Model for AMP Prediction using Group KAN Layers
# We modify a Vision Transformer model to use KAN layers with rational activations instead of standard MLPs.
# This implementation adapts the transformer architecture to handle AMP sequences.
class KATForAMP(VisionTransformer):
    def __init__(self, seq_length=100, embed_dim=768, num_classes=1, **kwargs):
        super().__init__(
            img_size=seq_length,  # Treat sequence length as the image size
            patch_size=1,  # Each amino acid is treated as an individual "patch"
            in_chans=1,  # Single integer channel per amino acid
            embed_dim=embed_dim,
            num_classes=num_classes,
            **kwargs,
        )

        # AMP Sequence Embedding
        # The model uses an embedding layer to represent amino acid sequences numerically.
        # This embedding layer maps each amino acid (26 in total) to a vector of `embed_dim` dimensions.
        self.embedding = nn.Embedding(26, embed_dim)  # 26 amino acids
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )  # Class token for sequence classification
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_length + 1, embed_dim)
        )  # Positional embedding for each token
        trunc_normal_(self.pos_embed, std=0.02)  # Initialize positional embedding

        # Replace MLP layers in transformer blocks with GroupKANLayer
        # Each transformer block is modified to replace the standard MLP with GroupKANLayer.
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": block.attn,  # Retain the original self-attention mechanism
                        "norm1": block.norm1,  # Layer normalization before self-attention
                        "group_kan": GroupKANLayer(
                            embed_dim, embed_dim
                        ),  # GroupKAN replaces standard MLPs
                        "norm2": block.norm2,  # Layer normalization before GroupKANLayer
                    }
                )
                for block in self.blocks
            ]
        )

        # Define classification head
        # The head layer projects the final class token embedding to a single output for binary classification.
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Custom initialization for model weights, similar to the KAT paper's approach
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)
        elif isinstance(m, GroupKANLayer):
            # Custom initialization for each group within GroupKANLayer
            for linear in m.linears:
                trunc_normal_(linear.weight, std=0.02)
                if linear.bias is not None:
                    nn.init.constant_(linear.bias, 0)

    def forward_features(self, x):
        # Embed the sequence data (numerical representation of amino acids)
        x = self.embedding(x)  # Shape: [batch_size, seq_length, embed_dim]

        # Add class token and positional embedding
        cls_tokens = self.cls_token.expand(
            x.size(0), -1, -1
        )  # Class token for each batch
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # Shape: [batch_size, seq_length + 1, embed_dim]

        # Adjust positional embeddings to match the input length
        pos_embed = self.pos_embed[
            :, : x.size(1), :
        ]  # Trim pos_embed to match input length
        x = x + pos_embed

        # Pass through the transformer blocks
        for block in self.blocks:
            # Self-attention layer
            x = block["attention"](block["norm1"](x)) + x
            # GroupKAN layer with rational activation
            x = block["group_kan"](block["norm2"](x)) + x

        # Final layer normalization on the class token
        x = self.norm(x)
        return x[
            :, 0
        ]  # Only the class token's representation is used for classification

    def forward(self, x):
        # Forward pass through features and classification head
        x = self.forward_features(x)
        x = self.head(x).squeeze(-1)  # Output shape: [batch_size]
        return x

    def save_weights_to_csv(self, filename="weights.csv"):
        # Collect weights from all linear layers in GroupKANLayer
        weights = {}
        for i, block in enumerate(self.blocks):
            for j, linear in enumerate(block["group_kan"].linears):
                weights[f"block_{i}_group_{j}_weights"] = (
                    linear.weight.data.cpu().numpy().flatten()
                )
                if linear.bias is not None:
                    weights[f"block_{i}_group_{j}_bias"] = (
                        linear.bias.data.cpu().numpy().flatten()
                    )

        # Convert weights dictionary to DataFrame and save to CSV
        weights_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in weights.items()]))
        weights_df.to_csv(filename, index=False)


# Instantiate the Model
# `create_kat_amp_model` initializes the model with the specified sequence length, embedding dimensions, and output classes.
def create_kat_amp_model(seq_length=100, embed_dim=768, num_classes=1):
    model = KATForAMP(
        seq_length=seq_length, embed_dim=embed_dim, num_classes=num_classes
    )
    return model
