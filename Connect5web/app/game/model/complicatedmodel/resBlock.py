from torch import nn
from convBlock import ConvBlock


class ResBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, in_channels, out_channels):
        """
        初始化。
        :param in_channels: 输入通道数。
        :param out_channels: 输出通道数。
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        """
        前馈
        :param x: 输入state
        :return: 经残差层处理过的state
        """
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x
