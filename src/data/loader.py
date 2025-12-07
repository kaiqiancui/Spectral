import torch
import pickle
import os
from .tasks import TaskFactory
from .dataset import UnifiedICLDataset

def load_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    print(f"🧠 Loading Cache: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def apply_cache(data_list, cache, key_name):
    """将数据列表中的文本替换为 Embedding"""
    valid_data = []
    miss_count = 0
    
    for item in data_list:
        text = item[key_name]
        if text in cache:
            # 复制一份 item 防止修改原始引用
            new_item = item.copy()
            emb = torch.tensor(cache[text])
            # 将 embedding 存入新的 key，例如 input1_emb
            new_item[f"{key_name}_emb"] = emb
            valid_data.append(new_item)
        else:
            miss_count += 1
            
    print(f"   Key: {key_name} | Hit: {len(valid_data)} | Miss: {miss_count}")
    return valid_data

def get_data_loader(cfg):
    # 1. 加载原始数据
    raw_splits = TaskFactory.load_raw_data(cfg.data.task_name, cfg)
    
    # 2. 遍历配置中的 inputs，加载对应的 Cache 并替换
    # cfg.data.inputs 是一个列表，包含 {column: 'input1', cache_path: '...'}
    for input_cfg in cfg.data.inputs:
        col_name = input_cfg.column
        cache_path = input_cfg.cache_path
        
        # 加载缓存
        cache = load_cache(cache_path)
        
        # 对 Train 和 Test 分别进行替换
        raw_splits['train'] = apply_cache(raw_splits['train'], cache, col_name)
        raw_splits['test'] = apply_cache(raw_splits['test'], cache, col_name)

    # 3. 封装成 PyTorch Dataset
    # 这里需要 Dataset 类支持字典列表的输入
    return {
        "train": UnifiedICLDataset(raw_splits['train'], cfg),
        "test": UnifiedICLDataset(raw_splits['test'], cfg)
    }
    
def collate_fn(batch_list):
    """
    batch_list: List[Dict], e.g., [{'input1_emb': Tensor, 'label': 1.2}, ...]
    """
    batch_out = {}
    
    # 假设所有样本的 keys 是一样的
    keys = batch_list[0].keys()
    
    for k in keys:
        values = [item[k] for item in batch_list]
        
        # 如果是 Tensor (Embedding)，stack 起来
        if isinstance(values[0], torch.Tensor):
            batch_out[k] = torch.stack(values)
        # 如果是数字 (Label)，转 Tensor
        elif isinstance(values[0], (int, float)):
            batch_out[k] = torch.tensor(values)
        # 否则 (字符串等)，保持 List
        else:
            batch_out[k] = values
            
    return batch_out