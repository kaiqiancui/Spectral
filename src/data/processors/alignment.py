import torch
import numpy as np
from .base import BaseProcessor

class AlignmentProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.target_dim = config.llm.hidden_size  # e.g. 4096
        self.source_dim = config.data.input_dim   # e.g. 640 or 1280
        
        # 1. 初始化随机投影矩阵 W (Frozen)
        # 对应原仓库的 Random Projection 逻辑
        self.projection_matrix = torch.randn(self.source_dim, self.target_dim)
        # 归一化投影矩阵，防止数值爆炸
        self.projection_matrix = self.projection_matrix / np.sqrt(self.target_dim)

    def fit_transform(self, train_emb, test_emb):
        """
        逻辑：
        1. 投影: X @ W
        2. 计算对齐统计量 (OT): 匹配 Train 的 Mean/Cov 到 LLM 的 Mean/Cov
        """
        # 转 Tensor
        if isinstance(train_emb, np.ndarray): train_emb = torch.from_numpy(train_emb).float()
        if isinstance(test_emb, np.ndarray): test_emb = torch.from_numpy(test_emb).float()

        # Step 1: Random Projection (Up-projection)
        # [N, 640] @ [640, 4096] -> [N, 4096]
        train_proj = train_emb @ self.projection_matrix
        test_proj = test_emb @ self.projection_matrix

        # Step 2: Optimal Transport (OT) / Gaussian Alignment
        # 原仓库的 "OT" 通常指对齐均值和方差 (Monge Map for Gaussians)
        # 目标统计量应来自 LLM Embedding (需要从 Config 或外部传入)
        target_mean = self.cfg.alignment.target_mean
        target_std = self.cfg.alignment.target_std

        # 计算当前的统计量
        src_mean = train_proj.mean(dim=0)
        src_std = train_proj.std(dim=0) + 1e-6

        # 应用对齐 (Z-score normalization + Rescale)
        # 这就是最基础的 OT (Diagonal Gaussian Alignment)
        train_aligned = (train_proj - src_mean) / src_std * target_std + target_mean
        test_aligned = (test_proj - src_mean) / src_std * target_std + target_mean

        return train_aligned, test_aligned