from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def render_gaussians(coords, size, sigma):
    """Differentiable-free Gaussian renderer, same convention as the dataset:
    pixel = x * (size - 1), truncated at 3*sigma, points outside [0,1] skipped.

    coords: (B, 4, 2) normalized. Returns (B, 4, size, size) float32.
    """
    coords = coords.float()
    grid = torch.arange(size, device=coords.device, dtype=torch.float32)
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))

    px = coords[..., 0] * (size - 1)
    py = coords[..., 1] * (size - 1)

    def axis(p):
        d = grid[None, None, :] - p[..., None]
        g = torch.exp(-(d**2) / (2.0 * sigma * sigma))
        base = torch.floor(p)[..., None]
        inside = (grid[None, None, :] >= base - radius) & (
            grid[None, None, :] <= base + radius
        )
        return g * inside

    gx = axis(px)
    gy = axis(py)

    valid = (
        (coords[..., 0] >= 0)
        & (coords[..., 0] <= 1)
        & (coords[..., 1] >= 0)
        & (coords[..., 1] <= 1)
    ).float()

    hm = gy[..., :, None] * gx[..., None, :]  # (B, 4, H, W)

    return hm * valid[..., None, None]


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class KeypointRefinementNet(nn.Module):
    """ResNet18 + FPN decoder + soft-argmax, optionally cascaded.

    Input : 7 channels (RGB + 4 initial-keypoint heatmaps).
    Output: heatmaps at stride 4, keypoints decoded by soft-argmax.
    `max_delta` is accepted only for config compatibility (unused).
    """

    def __init__(
        self,
        pretrained: bool = False,
        max_delta: float = 0.5,
        cond_dim: int = 0,
        cond_embed_dim: int = 64,
        num_iters: int = 2,
        input_sigma: float = 4.0,
        decoder_channels: int = 128,
    ):
        super().__init__()

        self.cond_dim = int(cond_dim)
        self.num_iters = int(num_iters)
        self.input_sigma = float(input_sigma)

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
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4,  64 ch
        self.layer2 = backbone.layer2  # stride 8,  128 ch
        self.layer3 = backbone.layer3  # stride 16, 256 ch
        self.layer4 = backbone.layer4  # stride 32, 512 ch

        c = decoder_channels

        self.red4 = ConvBNReLU(512, c)
        self.lat3 = nn.Conv2d(256, c, 1)
        self.lat2 = nn.Conv2d(128, c, 1)
        self.lat1 = nn.Conv2d(64, c, 1)

        self.smooth3 = ConvBNReLU(c, c)
        self.smooth2 = ConvBNReLU(c, c)
        self.smooth1 = ConvBNReLU(c, c)

        self.head = nn.Sequential(
            ConvBNReLU(c, 64),
            nn.Conv2d(64, 4, 1),
        )

        if self.cond_dim > 0:
            self.cond_encoder = nn.Sequential(
                nn.Linear(self.cond_dim, cond_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cond_embed_dim, cond_embed_dim),
                nn.ReLU(inplace=True),
            )
            self.cond_to_feat = nn.Linear(cond_embed_dim, c)
        else:
            self.cond_encoder = None
            self.cond_to_feat = None

    @staticmethod
    def _up_add(x, skip):
        return (
            F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            + skip
        )

    def _single(self, x, cond_vec):
        s = self.stem(x)
        x1 = self.layer1(s)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d = self.red4(x4)

        if cond_vec is not None:
            d = d + self.cond_to_feat(cond_vec)[:, :, None, None]

        d = self.smooth3(self._up_add(d, self.lat3(x3)))
        d = self.smooth2(self._up_add(d, self.lat2(x2)))
        d = self.smooth1(self._up_add(d, self.lat1(x1)))

        return self.head(d)  # (B, 4, H/4, W/4) logits

    @staticmethod
    def soft_argmax(logits):
        b, k, h, w = logits.shape

        p = F.softmax(logits.float().flatten(2), dim=-1).view(b, k, h, w)

        xs = torch.linspace(0.0, 1.0, w, device=logits.device)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device)

        x = (p.sum(dim=2) * xs).sum(dim=-1)
        y = (p.sum(dim=3) * ys).sum(dim=-1)

        return torch.stack([x, y], dim=-1)  # (B, 4, 2)

    def forward(
        self,
        image_and_heatmaps: torch.Tensor,
        initial_keypoints: torch.Tensor,
        cond: torch.Tensor | None = None,
        num_iters: int | None = None,
    ):
        n = self.num_iters if num_iters is None else int(num_iters)

        cond_vec = None

        if self.cond_encoder is not None:
            if cond is None:
                raise ValueError("Model was built with cond_dim > 0 but cond is None.")
            cond_vec = self.cond_encoder(cond)

        rgb = image_and_heatmaps[:, :3]
        size = image_and_heatmaps.shape[-1]

        kp = initial_keypoints.float()
        inp = image_and_heatmaps
        stages = []

        for i in range(max(1, n)):
            if i > 0:
                hm = render_gaussians(kp.detach(), size, self.input_sigma)
                inp = torch.cat([rgb, hm.to(rgb.dtype)], dim=1)

            logits = self._single(inp, cond_vec)
            kp = self.soft_argmax(logits)

            stages.append({"heatmaps": logits, "keypoints": kp})

        return {
            "heatmaps": stages[-1]["heatmaps"],
            "keypoints": kp,
            "delta": kp - initial_keypoints.float(),
            "stages": stages,
        }


def build_model_from_config(model_cfg, cond_dim, pretrained=False):
    """Build the right architecture from a (checkpoint) model config.

    A config without "arch" belongs to a checkpoint trained with the old
    pooled-regression network (models/keypoint_refinement_v1.py).
    """
    model_cfg = model_cfg or {}

    if model_cfg.get("arch") is None:
        from models.keypoint_refinement_v1 import (
            KeypointRefinementNet as LegacyNet,
        )

        return LegacyNet(
            pretrained=pretrained,
            max_delta=float(model_cfg.get("max_delta", 0.5)),
            cond_dim=int(cond_dim),
        )

    return KeypointRefinementNet(
        pretrained=pretrained,
        cond_dim=int(cond_dim),
        num_iters=int(model_cfg.get("num_iters", 2)),
    )
