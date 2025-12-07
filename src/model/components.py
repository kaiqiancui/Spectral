import torch
import torch.nn as nn

class Projector(nn.Module):
    """
    科学特征投影层：将 UniMol/ESM 的 Embedding 映射到 LLM 的 Hidden Size。
    架构参考原仓库：Linear -> GELU -> Linear
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def build_projector(config):
    """
    工厂函数，根据配置创建 Projector
    """
    return Projector(
        input_dim=config.get("science_dim", 1024), # 例如 UniMol 是 512, ESM 是 1280
        output_dim=config.get("llm_dim", 4096),     # Llama 3.1 8B 是 4096
        hidden_dim=config.get("projector_hidden_dim", 4096)
    )