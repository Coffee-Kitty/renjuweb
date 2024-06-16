import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        # 使用3x3的卷积核，步长为1，填充为1
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


# 搭建骨干网络，输入：N, 9, 10, 9 --> N, C, H, W
class Net(nn.Module):

    def __init__(self, num_channels=128, num_res_blocks=7,board_width=15, height=15):
        super().__init__()
        # 全局特征
        # self.global_conv = nn.Conv2D(in_channels=9, out_channels=512, kernel_size=(10, 9))
        # self.global_bn = nn.BatchNorm2D(512)
        # 初始化特征
        self.conv_block = nn.Conv2d(in_channels=4, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1)
        self.conv_block_bn = nn.BatchNorm2d(num_channels)
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 15 * 15, 15*15)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 15 * 15, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    # 定义前向传播
    def forward(self, x):
        x = x.reshape(-1, 4, 15, 15)
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16* 15 * 15])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy,dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 15 * 15])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value
#
# class PolicyValueNet():
#     """策略价值网络"""
#
#     def __init__(self, board_width, board_height,
#                  model_file=None, use_gpu=True):
#         self.use_gpu = use_gpu
#         self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
#         self.board_width = board_width
#         self.board_height = board_height
#         self.l2_const = 1e-3  # coef of l2 penalty
#
#         self.policy_value_net = Net(self.board_width, self.board_height).to(self.device)
#
#         self.optimizer = torch.optim.Adam(lr=0.02,
#                                           params=self.policy_value_net.parameters(),
#                                           weight_decay=self.l2_const)
#
#         if model_file:
#             net_params = torch.load(model_file)
#             self.policy_value_net.load_state_dict(net_params)
#
#     def policy_value(self, state_batch):
#         """
#        input: a batch of states
#        output: a batch of action probabilities and state values
#        """
#         self.policy_value_net.eval()
#
#         state_batch = torch.as_tensor(dtype=torch.float32, data=state_batch).to(self.device)
#         log_act_probs, values = self.policy_value_net(state_batch)
#         act_probs = np.exp(log_act_probs.detach().cpu().numpy())
#         return act_probs, values.detach().cpu().numpy()
#
#     def policy_value_fn(self, board):
#         """
#         input: board
#         output: a list of (action, probability) tuples for each available
#         action and the score of the board state
#         """
#         legal_positions = board.availables
#         # 数组在内存中存放的地址也是连续的
#         current_state = np.ascontiguousarray(board.current_state().reshape(
#             -1, 4, self.board_width, self.board_height)).astype("float32")
#         act_probs, value = self.policy_value(current_state)
#         act_probs = act_probs.flatten()
#         act_probs = zip(legal_positions, act_probs[legal_positions])
#         return list(act_probs), value
#
#     def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
#         """perform a training step"""
#         self.policy_value_net.train()
#
#         state_batch = torch.from_numpy(state_batch).to(self.device)
#         mcts_probs = torch.from_numpy(mcts_probs).to(self.device)
#         winner_batch = torch.from_numpy(winner_batch).to(self.device)
#
#         # zero the parameter gradients
#         self.optimizer.zero_grad()
#         # set learning rate
#         for params in self.optimizer.param_groups:
#             params['lr'] = lr
#
#         # forward
#         log_act_probs, value = self.policy_value_net(state_batch)
#         value = torch.reshape(value, shape=[-1])
#         value_loss = F.mse_loss(input=value, target=winner_batch)
#         policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
#         print(f"p_loss:{policy_loss},v_loss{value_loss}")
#         loss = value_loss + policy_loss
#         # backward and optimize
#         loss.backward()
#         self.optimizer.step()
#
#         # 计算策略的熵，仅用于评估模型
#         with torch.no_grad():
#             entropy = -torch.mean(
#                 torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
#             )
#         return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()
#
#     def get_policy_param(self):
#         net_params = self.policy_value_net.state_dict()
#         return net_params
#
#     def save_model(self, model_file):
#         """ save model params to file """
#         net_params = self.get_policy_param()  # get model params
#         torch.save(net_params, model_file)
#
#
# if __name__ == '__main__':
#     net = Net(board_width=15, height=15)
#     test_data = torch.ones((1, 4, 15, 15))
#     x_act, x_val = net(test_data)
#     print(x_act.shape)
#     print(x_val.shape)
#     policy_value = PolicyValueNet(board_width=15, board_height=15, model_file=None, use_gpu=True)
#     board = Board(width=15, height=15)
#     board.init_board()
#     act_proobs, v = policy_value.policy_value_fn(board)
#     print(act_proobs)
#     print(v)
#     state = board.current_state()
#     act_probs, v = policy_value.policy_value(state)
#     print(act_probs)
#     print(v)
