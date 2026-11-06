from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class KeypointRefinementNet(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        max_delta: float = 0.5,
        cond_dim: int = 0,
        cond_embed_dim: int = 64,
    ):
        super().__init__()

        self.max_delta = float(max_delta)
        self.cond_dim = int(cond_dim)

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            7,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = (
                old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1) * 0.25
            )

        backbone.conv1 = new_conv

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(16, 16), mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 4, 1),
        )

        if self.cond_dim > 0:
            self.cond_encoder = nn.Sequential(
                nn.Linear(self.cond_dim, cond_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cond_embed_dim, cond_embed_dim),
                nn.ReLU(inplace=True),
            )
            extra = cond_embed_dim
        else:
            self.cond_encoder = None
            extra = 0

        self.coordinate_head = nn.Sequential(
            nn.Linear(512 + 8 + extra, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
        )

    def forward(
        self,
        image_and_heatmaps: torch.Tensor,
        initial_keypoints: torch.Tensor,
        cond: torch.Tensor | None = None,
    ):
        x = self.stem(image_and_heatmaps)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        heatmaps = self.heatmap_head(x)

        features = self.avgpool(x).flatten(1)
        parts = [features, initial_keypoints.flatten(1)]

        if self.cond_encoder is not None:
            if cond is None:
                raise ValueError("Model was built with cond_dim > 0 but cond is None.")
            parts.append(self.cond_encoder(cond))

        delta = self.coordinate_head(torch.cat(parts, dim=1))
        delta = torch.tanh(delta) * self.max_delta
        delta = delta.view(-1, 4, 2)

        return {
            "heatmaps": heatmaps,
            "delta": delta,
            "keypoints": initial_keypoints + delta,
        }
