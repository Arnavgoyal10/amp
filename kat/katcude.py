import torch
import torch.nn as nn
from kat_rational import KAT_Group  # Importing KAT_Group from the repository
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_


# Group KAN Layer using KAT_Group for activation
class GroupKANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=None,
        bias=True,
        drop=0.0,
    ):
        super(GroupKANLayer, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define linear layers and KAT_Group activations
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias).cuda()
        self.act1 = (
            KAT_Group(mode=act_cfg["act_init"][0]).cuda()
            if act_cfg
            else KAT_Group().cuda()
        )
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias).cuda()
        self.act2 = (
            KAT_Group(mode=act_cfg["act_init"][1]).cuda()
            if act_cfg
            else KAT_Group().cuda()
        )
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x


# KAT Model for AMP Prediction using Group KAN Layers
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
        act_cfg = dict(
            type="KAT", act_init=["identity", "gelu"]
        )  # Configure activation types
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": block.attn,
                        "norm1": block.norm1,
                        "group_kan": GroupKANLayer(
                            embed_dim, hidden_features=embed_dim, act_cfg=act_cfg
                        ),
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
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)

    def forward_features(self, x):
        x = self.embedding(x).cuda()
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.blocks:
            x = block["attention"](block["norm1"](x)) + x
            x = block["group_kan"](block["norm2"](x)) + x

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x).squeeze(-1)


# Instantiate the KAT Model
def create_kat_amp_model(seq_length=100, embed_dim=768, num_classes=1):
    model = KATForAMP(
        seq_length=seq_length, embed_dim=embed_dim, num_classes=num_classes
    )
    return model
