# src/models/frame_selector.py
import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as tvm

class FrameSelectorModel(nn.Module):
    def __init__(
        self,
        bert_name: str = "vinai/phobert-base",
        vision_name: str = "mobilenet_v3_small",
        proj_dim: int = 256,
        freeze_text: bool = True,
        freeze_vision: bool = True,
    ):
        super().__init__()

        # -------- Vision backbone (pretrained) --------
        if vision_name == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = tvm.mobilenet_v3_small(weights=weights)
            # mobilenet forward: features -> avgpool -> flatten -> classifier
            # set classifier=Identity => returns pooled feature vector
            vision_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            self.vision = backbone
        elif vision_name == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            backbone = tvm.resnet18(weights=weights)
            vision_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.vision = backbone
        else:
            raise ValueError(f"Unsupported vision_name={vision_name}")

        # -------- Text backbone --------
        self.text = AutoModel.from_pretrained(bert_name)
        text_dim = self.text.config.hidden_size

        # -------- Projections --------
        self.v_proj = nn.Sequential(
            nn.Linear(vision_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
        )

        # temperature helps calibration of similarity logits
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # temp = exp(log_temp)

        # Freeze options
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text.parameters():
                p.requires_grad = False

    def forward(self, video, input_ids, attention_mask):
        """
        video: [B, N, C, H, W]
        input_ids: [B, L]
        attention_mask: [B, L]
        return logits: [B, N]  (NO sigmoid)
        """
        B, N, C, H, W = video.shape

        # vision feats
        x = video.view(B * N, C, H, W)
        v = self.vision(x)                 # [B*N, Dv]
        v = v.view(B, N, -1)               # [B, N, Dv]
        v = self.v_proj(v)                 # [B, N, d]

        # text feats (CLS)
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, Dt]
        t = self.t_proj(t)                 # [B, d]
        t = t.unsqueeze(1).expand(-1, N, -1)  # [B, N, d]

        # similarity logits
        temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
        logits = (v * t).sum(dim=-1) * temp  # [B, N]
        return logits
