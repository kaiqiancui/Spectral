# src/data/cache.py
import torch
import os
import pickle

class EmbeddingCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.data = self._load_cache()
        print(f"🧠 Embedding Cache Loaded. Size: {len(self.data)}")

    def _load_cache(self):
        # 兼容原仓库的 pickle 格式
        with open(self.cache_path, "rb") as f:
            return pickle.load(f)

    def get_batch(self, texts):
        """
        核心逻辑：输入文本列表，返回 Embedding Tensor 和 有效索引掩码
        """
        embs = []
        valid_indices = []
        
        for idx, text in enumerate(texts):
            # 原仓库逻辑是直接查字典，我们这里保持一致，不要搞正则清洗，确保 Cache 和 Raw Data 源头一致
            if text in self.data:
                embs.append(torch.tensor(self.data[text]))
                valid_indices.append(idx)
        
        if not embs:
            raise ValueError("❌ No cache hits! Check if your cache file matches the dataset.")
            
        return torch.stack(embs), valid_indices