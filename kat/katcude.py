import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_


# CUDA-Optimized Rational Activation Function
# This function approximates complex non-linear functions using rational functions,
# with CUDA-optimized computation.
class RationalActivation(nn.Module):
    def __init__(self):
        super(RationalActivation, self).__init__()
        # Numerator and denominator coefficients for rational activation function
        self.p = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"))
        self.q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0], device="cuda"))

    def forward(self, x):
        # Compute rational function: numerator / denominator
        numerator = self.p[0] + self.p[1] * x + self.p[2] * x**2 + self.p[3] * x**3
        denominator = 1 + torch.abs(self.q[0] * x + self.q[1] * x**2 + self.q[2] * x**3)
        return numerator / denominator


# Group KAN Layer with Rational Activations
# Implements grouped rational KAN layers for the KAT model, optimized for variance-preserving initialization.
class GroupKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_groups=8):
        super(GroupKANLayer, self).__init__()
        self.num_groups = num_groups
        group_in_features = in_features // num_groups
        group_out_features = out_features // num_groups

        # Initialize grouped linear layers and rational activations
        self.linears = nn.ModuleList(
            [
                nn.Linear(group_in_features, group_out_features).cuda()
                for _ in range(num_groups)
            ]
        )
        self.activations = nn.ModuleList(
            [RationalActivation() for _ in range(num_groups)]
        )

        # Custom variance-preserving initialization
        for linear in self.linears:
            trunc_normal_(linear.weight, std=0.02)
            nn.init.constant_(linear.bias, 0)
        for activation in self.activations:
            with torch.no_grad():
                activation.p.fill_(1.0)  # Initialize p coefficients
                activation.q.fill_(
                    0.1
                )  # Initialize q coefficients to stabilize denominator

    def forward(self, x):
        # Divide input into groups for group-wise transformations
        group_inputs = torch.chunk(x, self.num_groups, dim=-1)
        # Apply linear transformation and rational activation to each group
        group_outputs = [
            activation(linear(g_in))
            for g_in, linear, activation in zip(
                group_inputs, self.linears, self.activations
            )
        ]
        return torch.cat(group_outputs, dim=-1)


# KAT Model for AMP Prediction using Group KAN Layers
# This model modifies a Vision Transformer (ViT) to use grouped KAN layers with rational activations.
class KATForAMP(VisionTransformer):
    def __init__(self, seq_length=100, embed_dim=768, num_classes=1, **kwargs):
        super().__init__(
            img_size=seq_length,
            patch_size=1,
            in_chans=1,
            embed_dim=embed_dim,
            num_classes=num_classes,
            **kwargs
        )

        # AMP Sequence Embedding
        self.embedding = nn.Embedding(26, embed_dim).cuda()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device="cuda"))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_length + 1, embed_dim, device="cuda")
        )
        trunc_normal_(self.pos_embed, std=0.02)

        # Replace MLP layers in transformer blocks with GroupKANLayer
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": block.attn,
                        "norm1": block.norm1,
                        "group_kan": GroupKANLayer(embed_dim, embed_dim),
                        "norm2": block.norm2,
                    }
                )
                for block in self.blocks
            ]
        )

        # Classification head
        self.head = (
            nn.Linear(embed_dim, num_classes).cuda()
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Initialize weights, including custom variance-preserving for GroupKANLayer
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)
        elif isinstance(m, GroupKANLayer):
            for linear in m.linears:
                trunc_normal_(linear.weight, std=0.02)
                if linear.bias is not None:
                    nn.init.constant_(linear.bias, 0)

    def forward_features(self, x):
        # Embed sequence data for amino acids
        x = self.embedding(x).cuda()  # Shape: [batch_size, seq_length, embed_dim]
        cls_tokens = self.cls_token.expand(
            x.size(0), -1, -1
        )  # Class token for each batch
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # Shape: [batch_size, seq_length + 1, embed_dim]

        # Add positional embedding
        pos_embed = self.pos_embed[:, : x.size(1), :]
        x = x + pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block["attention"](block["norm1"](x)) + x
            x = block["group_kan"](block["norm2"](x)) + x

        # Layer normalization on the class token
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        # Forward pass through features and classification head
        x = self.forward_features(x)
        return self.head(x).squeeze(-1)


# Instantiate the KAT Model
def create_kat_amp_model(seq_length=100, embed_dim=768, num_classes=1):
    model = KATForAMP(
        seq_length=seq_length, embed_dim=embed_dim, num_classes=num_classes
    )
    return model
