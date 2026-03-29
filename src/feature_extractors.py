from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class InceptionV3Feature(nn.Module):
    """
    Approximate FID-style feature extractor using torchvision InceptionV3.
    This is not numerically identical to torch-fidelity's inception-v3-compat,
    but it follows the same general idea: resize -> normalize -> pool features.
    """
    def __init__(self):
        super().__init__()
        weights = tvm.Inception_V3_Weights.DEFAULT
        model = tvm.inception_v3(weights=weights, aux_logits=False, transform_input=False)
        model.fc = nn.Identity()
        self.model = model.eval()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [-1, 1] or roughly centered image space
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.model(x)   # [B, 2048]
        return feat


class DinoV2Feature(nn.Module):
    """
    Semantic feature extractor using timm DINOv2.
    Usually much better for visualization than raw-pixel PCA.
    """
    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        x = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.model(x)   # [B, D]
        return feat


def build_feature_extractor(
    name: Literal["inception", "dinov2"] = "dinov2",
) -> nn.Module:
    if name == "inception":
        return InceptionV3Feature()
    elif name == "dinov2":
        return DinoV2Feature()
    else:
        raise ValueError(f"Unknown feature extractor: {name}")