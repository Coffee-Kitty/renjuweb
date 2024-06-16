import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from app.game.backened.game import Board


class Net(nn.Module):
     """
     卷积神经网络
     """
     def __init__(self, board_width, height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = height
        # 通用层 common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # 行动策略层 action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * self.board_width * self.board_height,
                                 self.board_width * self.board_height)
        # 状态值层 state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * self.board_width * self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

     def forward(self, feature):
        # 通用层 common layers
        x = F.relu(self.conv1(feature))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        # 行动策略层 action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 状态值层 state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        # 输出行动可能性 和 终局的预期状态值
        return x_act, x_val
#
# class ResidualBlock(nn.Module):
#     """
#     残差块
#     """
#
#     def __init__(self, in_channel, out_channel):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),  # inplace=True 参数表示在原地进行操作
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channel))
#         self.right = nn.Sequential(
#             # Use Conv2d with the kernel_size of 1, without padding to improve the parameters of the network
#             nn.Conv2d(in_channel, out_channel, 1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channel))
#
#     def forward(self, x):
#         out = self.left(x)
#         residual = x if self.right is None else self.right(x)
#         out = out + residual
#         return F.relu(out)
#
#
# class Net(nn.Module):
#     def __init__(self, board_width, height):
#         super().__init__()
#         self.board_width = board_width
#         self.board_height = height
#         self.layer1 = ResidualBlock(4, 16)
#         self.layer2 = ResidualBlock(16, 32)
#         self.layer3 = ResidualBlock(32, 48)
#         # self.layer4 = ResidualBlock(32, 48)
#         # self.layer5 = ResidualBlock(64, 64)
#         # self.layer6 = ResidualBlock(64, 16)
#         # self.layer7 = ResidualBlock(128, 96)
#         # self.layer8 = ResidualBlock(96, 64)
#
#         # policy network
#         self.layer1_p = ResidualBlock(48, 64)
#         self.layer2_p = ResidualBlock(64, 16)
#         self.policy_fc = nn.Linear(16 * self.board_width * self.board_height, 512)
#         self.policy_batch_norm = nn.LayerNorm(512)
#         self.policy = nn.Linear(512, self.board_width * self.board_height)
#         # value network
#         self.layer1_v = ResidualBlock(48, 64)
#         self.layer2_v = ResidualBlock(64, 16)
#         self.value_fc = nn.Linear(16 * self.board_width * self.board_height, 64)
#         self.value_batch_norm = nn.LayerNorm(64)
#         self.value = nn.Linear(64, 1)
#
#     def forward(self, x):
#         x = x.reshape(-1, 4, self.board_width, self.board_height)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         # x = self.layer4(x)
#         # x = self.layer5(x)
#         # x = self.layer6(x)
#         # x = self.layer7(x)
#         # x = self.layer8(x)
#
#         # policy network
#         pi = self.layer1_p(x)
#         pi = self.layer2_p(pi)
#         pi = pi.view(-1, 16 * self.board_width * self.board_height)
#         pi = self.policy_fc(pi)
#         pi = F.relu(self.policy_batch_norm(pi))
#         pi = F.log_softmax(self.policy(pi), dim=1)
#         # value network
#         v = self.layer1_v(x)
#         v = self.layer2_v(v)
#         v = v.view(-1, 16 * self.board_width * self.board_height)
#         v = self.value_fc(v)
#         v = F.relu(self.value_batch_norm(v))
#         v = torch.tanh(self.value(v))
#         return pi, v


class PolicyValueNet():
    """策略价值网络"""

    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True,best_model=None):
        self.use_gpu = use_gpu
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3  # coef of l2 penalty

        self.policy_value_net = Net(self.board_width, self.board_height)

        self.optimizer = torch.optim.Adam(lr=0.02,
                                          params=self.policy_value_net.parameters(),
                                          weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

        if best_model:
            self.policy_value_net=best_model
        self.policy_value_net.to(self.device)

    def policy_value(self, state_batch):
        """
       input: a batch of states
       output: a batch of action probabilities and state values
       """
        self.policy_value_net.eval()

        state_batch = torch.as_tensor(dtype=torch.float32, data=state_batch).to(self.device)
        log_act_probs, values = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, values.detach().cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        # 数组在内存中存放的地址也是连续的
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 3, self.board_width, self.board_height)).astype("float32")
        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs.flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return list(act_probs), value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """perform a training step"""
        self.policy_value_net.train()

        state_batch = torch.from_numpy(state_batch).to(self.device)
        mcts_probs = torch.from_numpy(mcts_probs).to(self.device)
        winner_batch = torch.from_numpy(winner_batch).to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        for params in self.optimizer.param_groups:
            params['lr'] = lr

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        value_loss = F.mse_loss(input=value, target=winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        print(f"p_loss:{policy_loss},v_loss{value_loss}")
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


if __name__ == '__main__':
    net = Net(board_width=15,height=15)
    test_data = torch.ones((2,4,15,15))
    x_act, x_val = net(test_data)
    print(x_act.shape)
    print(x_val.shape)
    policy_value = PolicyValueNet(board_width=15, board_height=15, model_file=None, use_gpu=True)
    board = Board(width=15, height=15)
    board.init_board()
    act_proobs, v = policy_value.policy_value_fn(board)
    print(act_proobs)
    print(v)
    state = board.current_state()
    act_probs, v = policy_value.policy_value(state)
    print(act_probs)
    print(v)
