import torch
import torch.nn as nn

class RandomProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 原论文使用 MLP_Linear，其实就是一个简单的线性映射
        # 关键：不需要复杂的激活函数，仅仅是维度变换
        self.net = nn.Linear(input_dim, output_dim)
        
        # 随机初始化 (He Initialization or Xavier)
        nn.init.kaiming_normal_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        
    def forward(self, x):
        return self.net(x)