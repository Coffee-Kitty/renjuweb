from convBlock import ConvBlock
from torch.nn import functional as F
import torch.nn as nn


class PolicyHead(nn.Module):
    """
    策略头
    """

    def __init__(self, in_channels=128):
        """
        初始化。
        :param in_channels: 输入通道数。
        :param board_len: 棋盘长度。
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=1,
                              kernel_size=1,
                              padding=0)
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        x = x.flatten(1)
        return x