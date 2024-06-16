import torch
from torch import nn

from convBlock import ConvBlock
from resBlock import ResBlock
from valueHead import ValueHead
from policyHead import PolicyHead


class Model(nn.Module):
    """
    策略价值网络。
    """

    def __init__(self, n_feature_planes: int,
                 num_block: int,
                 in_channels: int,
                 out_channels: int,
                 is_use_gpu=True):
        """
        初始化Model
        :param n_feature_planes: board输入特征数
        :param in_channels: 残差输入通道数
        :param out_channels: 残差输出通道数
        :param is_use_gpu: 是否使用GPU
        """
        super().__init__()
        self.is_use_gpu = is_use_gpu
        self.n_feature_planes = n_feature_planes
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
        self.conv = ConvBlock(in_channels=n_feature_planes,
                              out_channels=in_channels,
                              kernel_size=3,
                              padding=1)
        # num_block个残差块
        self.residues = nn.Sequential(*[ResBlock(in_channels, out_channels) for i in range(num_block)])
        self.value_head = ValueHead(in_channels=out_channels)
        self.policy_head = PolicyHead(in_channels=in_channels)

    def forward(self, x):
        """
        前馈
        :param x:输入的state-->(N, C, H, W)
        :return: 策略和价值
        """
        x = self.conv(x)
        x = self.residues(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        v = torch.softmax(v, dim=1)
        # print(sum(v[0]))
        value = v[:, 0] - v[:, 1]
        policy = torch.log_softmax(p, dim=1)
        # # policy = torch.exp(policy)
        return policy, value


if __name__ == '__main__':
    model = Model(2, 4, 32, 32)
    x = torch.zeros(size=(5, 2, 11, 11))
    p, v = model(x)
    print(v.shape, p.shape)

# RecursiveScriptModule(
#   original_name=PolicyValueNet
#   (conv): RecursiveScriptModule(
#     original_name=ConvBlock
#     (conv): RecursiveScriptModule(original_name=Conv2d)
#     (batch_norm): RecursiveScriptModule(original_name=BatchNorm2d)
#   )
#   (residues): RecursiveScriptModule(
#     original_name=Sequential
#     (0): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (1): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (2): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (3): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (4): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (5): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (6): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (7): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (8): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (9): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (10): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (11): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (12): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (13): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (14): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (15): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (16): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (17): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (18): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (19): RecursiveScriptModule(
#       original_name=ResidueBlock
#       (conv1): RecursiveScriptModule(original_name=Conv2d)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm1): RecursiveScriptModule(original_name=BatchNorm2d)
#       (batch_norm2): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#   )
#   (policy_head): RecursiveScriptModule(
#     original_name=PolicyHead
#     (conv): RecursiveScriptModule(
#       original_name=ConvBlock
#       (conv): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#   )
#   (value_head): RecursiveScriptModule(
#     original_name=ValueHead
#     (conv): RecursiveScriptModule(
#       original_name=ConvBlock
#       (conv): RecursiveScriptModule(original_name=Conv2d)
#       (batch_norm): RecursiveScriptModule(original_name=BatchNorm2d)
#     )
#     (fc): RecursiveScriptModule(
#       original_name=Sequential
#       (0): RecursiveScriptModule(original_name=Linear)
#       (1): RecursiveScriptModule(original_name=ReLU)
#       (2): RecursiveScriptModule(original_name=Linear)
#       (3): RecursiveScriptModule(original_name=Tanh)
#     )
#   )
# )