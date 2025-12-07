import os
import torch
import numpy as np
import ot  # 需要安装 POT 库: pip install POT
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import List, Dict, Any
from src.data.cache import CacheManager # 假设你有一个缓存管理器，如果没有，下文有简易实现

class PCAOTProcessor:
    def __init__(self, config, tokenizer):
        """
        处理基于 Embedding 的 OT-ICL 数据构建
        
        Args:
            config: Hydra 配置对象，包含 pca_dim, ot_epsilon, k_shot 等参数
            tokenizer: Llama Tokenizer
        """
        self.config = config
        self.tokenizer = tokenizer
        self.cache_dir = config.get("cache_dir", "./cache_ot")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_embeddings(self, file_path: str) -> np.ndarray:
        """加载预训练的科学特征 (UniMol/ESM)"""
        print(f"Loading embeddings from {file_path}...")
        data = torch.load(file_path, map_location='cpu')
        
        # 兼容原仓库可能的格式，可能是 dict 也可能是 tensor
        if isinstance(data, dict):
            # 假设键名可能是 'embeddings', 'feats', 或者直接是 cls token
            if 'embeddings' in data:
                return data['embeddings'].numpy()
            elif 'cls_repr' in data: # UniMol 常见键名
                return data['cls_repr'].numpy()
            else:
                raise ValueError(f"Unknown embedding format in {file_path}. Keys: {data.keys()}")
        elif isinstance(data, torch.Tensor):
            return data.numpy()
        else:
            return data

    def compute_pca(self, train_emb: np.ndarray, test_emb: np.ndarray, n_components: int = 20):
        """
        复刻原仓库逻辑：在训练集上Fit PCA，转换训练集和测试集
        """
        print(f"Running PCA (dim={n_components})...")
        pca = PCA(n_components=n_components)
        train_pca = pca.fit_transform(train_emb)
        test_pca = pca.transform(test_emb)
        return train_pca, test_pca

    def compute_ot_retrieval(self, 
                             train_emb_pca: np.ndarray, 
                             test_emb_pca: np.ndarray, 
                             k_shot: int,
                             reg: float = 0.1) -> torch.Tensor:
        """
        核心 OT 逻辑：计算 Sinkhorn 距离并获取检索索引
        """
        num_train = train_emb_pca.shape[0]
        num_test = test_emb_pca.shape[0]

        print(f"Computing Optimal Transport Matrix ({num_test}x{num_train})... This may take a while.")
        
        # 1. 计算 Cost Matrix (欧氏距离的平方)
        # 使用 POT 库的 dist 函数，或者手动 scipy.spatial.distance.cdist
        M = ot.dist(test_emb_pca, train_emb_pca, metric='euclidean')
        
        # 归一化 Cost Matrix (原仓库常用技巧，防止数值溢出)
        M /= M.max()

        # 2. 定义分布 (Uniform distribution)
        a = np.ones((num_test,)) / num_test
        b = np.ones((num_train,)) / num_train

        # 3. 求解 Sinkhorn
        # reg 是正则化参数 (entropy regularization)，对应原代码的 ot_epsilon
        try:
            ot_plan = ot.sinkhorn(a, b, M, reg)
        except Exception as e:
            print(f"Sinkhorn failed, falling back to EMD or checking params: {e}")
            # 如果 Sinkhorn 失败（如 NaN），有些代码会回退到 EMD，或者调整 reg
            # 这里简单处理，抛出异常或尝试更稳健的求解
            raise e

        # 4. 根据 Transport Plan 获取 Top-K 索引
        # ot_plan[i, j] 表示测试样本 i 传输到训练样本 j 的概率质量
        # 我们取每行最大的 K 个值的索引
        
        # 转换为 PyTorch 处理
        ot_plan_tensor = torch.from_numpy(ot_plan)
        
        # 对于每个测试样本，找到权重最大的 K 个训练样本索引
        _, top_k_indices = torch.topk(ot_plan_tensor, k=k_shot, dim=1)
        
        return top_k_indices

    def get_icl_indices(self, train_emb, test_emb, modality_name="molecule"):
        """
        获取 ICL 索引的主控函数，包含缓存逻辑
        """
        cache_name = f"{modality_name}_pca{self.config.pca_dim}_reg{self.config.ot_epsilon}_k{self.config.k_shot}_indices.pt"
        cache_path = os.path.join(self.cache_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"Loading cached OT indices from {cache_path}")
            return torch.load(cache_path)
        
        # 1. PCA
        train_pca, test_pca = self.compute_pca(train_emb, test_emb, n_components=self.config.pca_dim)
        
        # 2. OT Calculation
        top_k_indices = self.compute_ot_retrieval(train_pca, test_pca, 
                                                  k_shot=self.config.k_shot, 
                                                  reg=self.config.ot_epsilon)
        
        # 3. Save Cache
        torch.save(top_k_indices, cache_path)
        print(f"Saved OT indices to {cache_path}")
        
        return top_k_indices

    def process(self, train_data: List[Dict], test_data: List[Dict], 
                train_emb_path: str, test_emb_path: str, modality: str = "molecule"):
        """
        主处理函数
        Args:
            train_data: 包含文本信息的训练集列表 [{'text': '...', 'label': '...'}, ...]
            test_data: 包含文本信息的测试集列表
            train_emb_path: 训练集 Embedding 文件路径
            test_emb_path: 测试集 Embedding 文件路径
            modality: 'molecule' or 'protein'
        Returns:
            processed_test_data: 包含 input_ids (Text) 和 feature_embeds (Science) 的列表
        """
        
        # 1. 加载科学特征
        train_emb = self.load_embeddings(train_emb_path)
        test_emb = self.load_embeddings(test_emb_path)
        
        assert len(train_data) == len(train_emb), f"Train data size {len(train_data)} != Embedding size {len(train_emb)}"
        assert len(test_data) == len(test_emb), f"Test data size {len(test_data)} != Embedding size {len(test_emb)}"

        # 2. 获取检索索引 (Indices)
        # 这一步包含了 PCA -> OT 的所有计算
        retrieved_indices = self.get_icl_indices(train_emb, test_emb, modality_name=modality)

        processed_dataset = []

        # 3. 构建 Prompt
        print("Constructing ICL Prompts...")
        for i, test_item in tqdm(enumerate(test_data), total=len(test_data)):
            # 获取当前测试样本检索到的 K 个训练样本索引
            indices = retrieved_indices[i].tolist()
            
            # 拼接 Context (Demonstrations)
            context_str = ""
            for idx in indices:
                train_item = train_data[idx]
                # 这里的 prompt 格式需要根据你的 dataset.py 或 baseline.yaml 中的模板来定
                # 下面是一个通用的 "Q: ... A: ..." 格式，你需要根据实际任务修改
                # 原仓库逻辑：Question + Answer
                context_str += f"Question: {train_item['input']}\nAnswer: {train_item['output']}\n\n"
            
            # 拼接当前查询
            # 注意：原仓库是将 Projector 输出的 embedding 放在 Prompt 最前面
            # 所以这里的 text_input 主要负责文本部分
            current_query = f"Question: {test_item['input']}\nAnswer:"
            full_text = context_str + current_query
            
            # 4. 组装返回对象
            # 这里的 feature_emb 将被送入 Projector
            feature_emb = torch.tensor(test_emb[i], dtype=torch.float32) 
            
            processed_dataset.append({
                "raw_text": full_text,           # 用于 debug
                "feature_embeds": feature_emb,   # 科学模态特征 [1024]
                "target_text": test_item['output'] # Label
            })
            
        return processed_dataset