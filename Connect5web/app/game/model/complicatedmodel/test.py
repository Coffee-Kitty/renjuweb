import torch

from model import Model

"""
返回1*3*15*15的张量

// 1 当前玩家的棋子分布 有子为1 没有为0
// 2 对方的棋子分布 有子为1 没有为0
// 3 我方为黑方  全为-1  我方为白方 全为1
"""
path = '../20b128c_renju.pt'
model = torch.jit.load(path, map_location=torch.device('cuda'))

x = torch.rand(1,3,15,15).cuda()  # this is the batch tensors of detect work
pre = model(x)
print(pre[0],pre[1])

## 保存模型
torch.save({'model': model.state_dict()}, 'model_name.pth')
# print(model)
for param_tensor in model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())

## 读取模型
model = Model(3,20,128,128)
state_dict = torch.load('model_name.pth')
model.load_state_dict(state_dict['model'])

# path = "a.pt"
# model = torch.jit.load(path, map_location=torch.device('cuda'))
#
# x = torch.rand(1,3,15,15).cuda()  # this is the batch tensors of detect work
# pre = model(x)
# print(pre[0],pre[1])