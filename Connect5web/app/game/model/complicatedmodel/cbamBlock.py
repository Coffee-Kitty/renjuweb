import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            # 3x3 卷积实现空间注意力
            nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力加权
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        # 空间注意力加权
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return x

if __name__ == '__main__':
    model = CBAMBlock(32)
    state = torch.zeros(size=(1, 32, 11, 11))
    out = model(state)
    print(out.shape)