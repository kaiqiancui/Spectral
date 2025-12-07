import numpy as np
import torch
from scipy.fftpack import dct
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph
from .base import BaseProcessor

class SpectralProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.target_dim = self.cfg.method.params.target_dim
        self.params = self.cfg.method.params 
    def fit_transform(self, train_emb, test_emb):
        # 确保输入是 numpy
        if isinstance(train_emb, torch.Tensor):
            train_emb = train_emb.detach().cpu().numpy()
        if isinstance(test_emb, torch.Tensor):
            test_emb = test_emb.detach().cpu().numpy()
            
        print(f"[Spectral] Input shape: {train_emb.shape}")
        
        # =========================================
        # 1. [创新点] Fiedler Vector 重排序
        # =========================================
        if self.cfg.method.params.reorder:
            print("[Spectral] Computing Fiedler vector for reordering...")
            # 构建 KNN 图 (只在训练集上构建结构！)
            # 注意：这里的逻辑是对 Embedding 的 "特征维度(Dim)" 进行重排，还是对 "样本(N)" 进行重排？
            # 你的论文如果是做 Spectral-ICRL，通常是对 *Embedding 的特征维度* 进行排序，
            # 以便 DCT 能更好压缩。如果是这样，我们需要转置。
            
            # 假设 train_emb 是 [N_samples, D_features]
            # 我们想让 D_features 变得平滑，所以要计算 DxD 的相关性图
            X_feat = train_emb.T  # [D, N]
            
            # 为了节省计算，如果 D 很大 (如 4096)，可以用部分样本估计
            # 或者直接计算相关系数矩阵
            dist_matrix = np.corrcoef(X_feat) # [D, D]
            # 转换为邻接矩阵 (简单阈值或KNN)
            A = (np.abs(dist_matrix) > 0.5).astype(float) 
            np.fill_diagonal(A, 0)
            
            # 拉普拉斯矩阵
            D_mat = np.diag(A.sum(axis=1))
            L = D_mat - A
            
            # 特征分解，取第二小特征值对应的特征向量 (Fiedler Vector)
            # 使用 eigh 处理对称矩阵更快
            vals, vecs = eigh(L, subset_by_index=[1, 1])
            fiedler = vecs[:, 0]
            
            # 得到排序索引
            perm_idx = np.argsort(fiedler)
            
            # 应用重排序到特征维度
            train_emb = train_emb[:, perm_idx]
            test_emb = test_emb[:, perm_idx]
            print(f"[Spectral] Reordering applied. Top 5 indices: {perm_idx[:5]}")

        # =========================================
        # 2. DCT 变换 (离散余弦变换)
        # =========================================
        print("[Spectral] Applying DCT...")
        # axis=1 表示对特征维度进行变换
        train_dct = dct(train_emb, axis=1, norm='ortho')
        test_dct = dct(test_emb, axis=1, norm='ortho')

        # =========================================
        # 3. 截断与残差
        # =========================================
        k = self.target_dim
        # 如果需要留空间给残差，主要特征就少取一点
        k_main = k - 1 if self.cfg.method.params.add_residual else k
        
        train_final = train_dct[:, :k_main]
        test_final = test_dct[:, :k_main]
        
        if self.cfg.method.params.add_residual:
            print("[Spectral] Adding High-Frequency Residuals...")
            # 计算剩余高频部分的能量 (Variance)
            train_res = np.var(train_dct[:, k_main:], axis=1, keepdims=True)
            test_res = np.var(test_dct[:, k_main:], axis=1, keepdims=True)
            
            # 拼接
            train_final = np.concatenate([train_final, train_res], axis=1)
            test_final = np.concatenate([test_final, test_res], axis=1)

        # =========================================
        # 4. 谱整形 (Alignment to LLM Stats)
        # =========================================
        print("[Spectral] Aligning stats to LLM distribution...")
        # 目标统计量 (从 Config 读取)
        tgt_mean = self.cfg.method.params.align_stats.mean
        tgt_std = self.cfg.method.params.align_stats.std
        
        # 计算当前统计量 (只用 Train!)
        curr_mean = np.mean(train_final)
        curr_std = np.std(train_final)
        
        # 简单的 Z-score + Scale + Shift
        train_final = (train_final - curr_mean) / (curr_std + 1e-8) * tgt_std + tgt_mean
        test_final = (test_final - curr_mean) / (curr_std + 1e-8) * tgt_std + tgt_mean
        
        print(f"[Spectral] Final output shape: {train_final.shape}")
        return torch.tensor(train_final, dtype=torch.float32), torch.tensor(test_final, dtype=torch.float32)
