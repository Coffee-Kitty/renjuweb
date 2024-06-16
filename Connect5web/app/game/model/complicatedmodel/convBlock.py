import torch
from torch import nn
from torch.nn import functional as F
# from .cbamBlock import CBAMBlock
from seBlock import SEModule
# from .ecaBlock import ECAModule


class ConvBlock(nn.Module):
    """
    卷积块。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding=0):
        """
        初始化模块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param padding: 填充
        """
        super().__init__()
        # 卷积层
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        # 标准化,将输出的均值控制为0，方差控制为1
        self.cbam = SEModule(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        前馈
        :param x: state
        :return: 卷积后处理后的state
        """
        return F.relu(self.batch_norm(self.cbam(self.conv(x))))

if __name__ == '__main__':
    model = ConvBlock(2, 32, 3, 1)
    state = torch.zeros(size=(1, 2, 11, 11))
    out = model(state)
    print(out.shape)

