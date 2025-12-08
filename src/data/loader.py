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

def get_data_loader(config):
    # 获取真实数据
    fm_reps, labels, smiles = get_real_chem_data(config)
    
    # 构建 Dataset
    # 必须传入所有数据以便做 Stratified Sampling
    full_dataset = {
        'features': fm_reps,
        'labels': labels,
        'text': smiles
    }
    
    # 分割 Train/Test (需复刻原仓库的 random_split 或 stratified split)
    # ...
    
    return {
        "test": ICRLDataset(test_indices, full_dataset, tokenizer, config)
    }