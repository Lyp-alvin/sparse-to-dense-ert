from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = filters
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = ConvBlock(in_channels, f0)
        self.conv1_0 = ConvBlock(f0, f1)
        self.conv2_0 = ConvBlock(f1, f2)
        self.conv3_0 = ConvBlock(f2, f3)
        self.conv4_0 = ConvBlock(f3, f4)

        self.conv0_1 = ConvBlock(f0 + f1, f0)
        self.conv1_1 = ConvBlock(f1 + f2, f1)
        self.conv2_1 = ConvBlock(f2 + f3, f2)
        self.conv3_1 = ConvBlock(f3 + f4, f3)

        self.conv0_2 = ConvBlock(f0 * 2 + f1, f0)
        self.conv1_2 = ConvBlock(f1 * 2 + f2, f1)
        self.conv2_2 = ConvBlock(f2 * 2 + f3, f2)

        self.conv0_3 = ConvBlock(f0 * 3 + f1, f0)
        self.conv1_3 = ConvBlock(f1 * 3 + f2, f1)

        self.conv0_4 = ConvBlock(f0 * 4 + f1, f0)
        self.final = nn.Conv2d(f0, out_channels, kernel_size=1)

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample_like(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample_like(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample_like(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample_like(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample_like(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample_like(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample_like(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample_like(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample_like(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_like(x1_3, x0_0)], dim=1))
        return self.final(x0_4)
