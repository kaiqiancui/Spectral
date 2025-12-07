import torch
from torch.utils.data import Dataset
import random
import os

class UnifiedICLDataset(Dataset):
    def __init__(self, data_path, n_shots=0, mode='train'):
        """
        Args:
            data_path: preprocess.py 生成的 .pt 文件路径
            n_shots: ICL 示例数量
            mode: 'train' 或 'test'
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}")
            
        print(f"📂 Loading {mode} data from {data_path}...")
        data_bundle = torch.load(data_path)
        
        self.raw_data = data_bundle[mode]
        self.texts = self.raw_data['text']
        self.embeddings = self.raw_data['emb']  # [N, 640] 原始维度
        self.labels = self.raw_data['label']
        
        # Shot 来源永远是训练集
        self.shot_source = data_bundle['train']
        self.shot_indices = list(range(len(self.shot_source['text'])))
        
        self.n_shots = n_shots
        self.mode = mode
        
        print(f"✅ Loaded {len(self.texts)} samples.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1. Query
        query = {
            'text': self.texts[idx],
            'emb': self.embeddings[idx], 
            'label': float(self.labels[idx])
        }
        
        # 2. Shots (从训练集随机采样)
        shots = []
        if self.n_shots > 0:
            candidates = self.shot_indices.copy()
            # 训练时排除自己防止泄露
            if self.mode == 'train' and idx in candidates:
                candidates.remove(idx)
            
            # 随机采样
            selected = random.sample(candidates, min(self.n_shots, len(candidates)))
            
            for si in selected:
                shots.append({
                    'text': self.shot_source['text'][si],
                    'emb': self.shot_source['emb'][si],
                    'label': float(self.shot_source['label'][si])
                })
        
        return {'query': query, 'shots': shots}

def collate_fn(batch):
    query_texts = [b['query']['text'] for b in batch]
    query_embs = torch.stack([b['query']['emb'] for b in batch])
    query_labels = torch.tensor([b['query']['label'] for b in batch], dtype=torch.float32)
    shots_batch = [b['shots'] for b in batch]
    
    return {
        'net_input': {
            'query_text': query_texts,
            'query_emb': query_embs,
            'shots': shots_batch
        },
        'target': query_labels
    }