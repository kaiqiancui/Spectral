# 修改文件: Spectral-master/src/data/loader.py

import torch
import numpy as np
from src.data.dataset import ICRLDataset

def get_real_chem_data(config):
    """
    加载原仓库预处理好的 Tensor 文件
    """
    task_name = config.data.task_name # e.g. "ESOL"
    base_path = config.data.data_path # 指向原仓库的 datasets 目录
    
    print(f"Loading real chemistry data for {task_name}...")
    
    # 1. 加载 Label 和 SMILES (可以使用 deepchem 或直接读取 csv)
    # 这里建议直接读取原仓库处理好的 .csv 或 .json
    # ... (省略读取 csv 的代码)
    
    # 2. 加载 Cached Embeddings (最重要!)
    # 原仓库文件名为: smile_to_rep_store.pkl 或 特定 task 的 .pt
    cache_path = f"{base_path}/{task_name}_fm_reps_tensor_train.pt"
    
    if os.path.exists(cache_path):
        fm_reps = torch.load(cache_path)
    else:
        raise FileNotFoundError(f"找不到预计算的特征文件: {cache_path}，请先运行原仓库的 get_cache 脚本。")
        
    return fm_reps, labels, smiles

def get_data_loader(cfg):
    print(f"Loading embeddings from {cfg.data.train_emb_path}")
    # 假设你已经把原仓库跑出来的 Tensor 存成了 .pt
    train_embs = torch.load(cfg.data.train_emb_path, map_location='cpu')
    
    # 加载原始数据 (JSON/CSV)
    # 原仓库是分开加载的，这里需要小心顺序
    # 强烈建议: 不要分开加载。应该在原仓库做一个脚本，把 (smiles, label, embedding) 存成一个单一的 list of dicts 的 .pt 文件
    # 如果必须分开加载，必须确保这里的 datasets 加载顺序与生成 embedding 时的顺序完全一致！
    
    raw_data = load_raw_data_from_json(cfg.data.train_data_path) # 自定义函数读取json
    
    if len(raw_data) != len(train_embs):
        raise ValueError(f"数据量不匹配! Raw: {len(raw_data)}, Emb: {len(train_embs)}")
    
    # 组装
    full_dataset = []
    for i, item in enumerate(raw_data):
        full_dataset.append({
            "input_text": item['Drug'] if 'Drug' in item else item['smiles'], # 适配不同数据集key
            "label": item['Y'] if 'Y' in item else item['label'],
            "feature_emb": train_embs[i]
        })
    
    # 分割 Train/Test (需复刻原仓库的 random_split 或 stratified split)
    # ...
    
    return {
        "test": ICRLDataset(test_indices, full_dataset, tokenizer, config)
    }