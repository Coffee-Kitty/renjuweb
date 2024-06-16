import torch.nn as nn


class ValueHead(nn.Module):
    """ 价值头 """

    def __init__(self, in_channels=128):
        """
        价值头
        :param in_channels:
        """
        super().__init__()
        self.in_channels = in_channels
        self.vNet = nn.Linear(in_features=in_channels, out_features=3)

    def forward(self, x):
        value = x.mean((2, 3))
        value = self.vNet(value)
        return value