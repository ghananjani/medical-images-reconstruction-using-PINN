"""Small trainable 2D U-Net for CT FBP post-processing (not a PINN)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.GroupNorm(min(4, outputs), outputs),
            nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(min(4, outputs), outputs),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CTUNet(nn.Module):
    """Two-level encoder-decoder with skip connections and bounded output.

    Expects [B,1,H,W] with both spatial dimensions divisible by four.
    Input is an FBP reconstruction; output approximates a clean unit-scale image.
    """

    def __init__(self, base_channels: int = 8):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be divisible by four and >=4")
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, 2 * b)
        self.bridge = ConvBlock(2 * b, 4 * b)
        self.dec2 = ConvBlock(6 * b, 2 * b)
        self.dec1 = ConvBlock(3 * b, b)
        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or any(n % 4 for n in x.shape[-2:]):
            raise ValueError("expected [B,1,H,W], with H and W divisible by four")
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.bridge(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear",
                                                align_corners=False), b], dim=1))
        d = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="bilinear",
                                                align_corners=False), a], dim=1))
        return torch.sigmoid(self.output(d))
