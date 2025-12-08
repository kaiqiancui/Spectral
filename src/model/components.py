# 修改文件: Spectral-master/src/model/components.py

import torch
import torch.nn as nn

class MLP_Linear(nn.Module):
    """
    复刻原仓库 chemistry/utils/models.py 中的 MLP_Linear
    用于 Training-Free 或简单的线性对齐
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=bias)
        self.init_weights()

    def forward(self, x):
        return self.layer(x)

    def init_weights(self):
        # 原仓库的初始化逻辑：Normal(0, 0.02)
        init_std = 0.02
        self.layer.weight.data.normal_(0, init_std)
        if self.layer.bias is not None:
            self.layer.bias.data.fill_(0)

def build_projector(config):
    projector_type = config.get("projector_type", "linear")
    
    if projector_type == "linear":
        return MLP_Linear(
            input_dim=config.get("science_dim", 512),
            output_dim=config.get("llm_dim", 4096)
        )
    # 保留原本的复杂 Projector 作为可选项
    elif projector_type == "mlp":
        return nn.Sequential(
            nn.Linear(config.get("science_dim"), config.get("projector_hidden_dim")),
            nn.GELU(),
            nn.Linear(config.get("projector_hidden_dim"), config.get("llm_dim"))
        )
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")