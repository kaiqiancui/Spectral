import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.utils.esm_utils import esm_model 
from src.data.utils.utils import (
    load_DTI, load_ESOL, load_Stability, load_Fluorescence, 
    load_Beta_Lactamase, load_ppi_affinity
)

def _find_embedding_cache(data_root, config_emb_folder=None):
    filename = "aaseq_to_rep_store.pkl"
    search_dirs = []
    if config_emb_folder: search_dirs.append(config_emb_folder)
    if data_root:
        search_dirs.append(os.path.join(data_root, "embeddings"))
        search_dirs.append(os.path.join(data_root, "cache"))
        search_dirs.append(data_root)
    
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            # [核心修复] 转为绝对路径！
            abs_path = os.path.abspath(candidate)
            print(f"✅ Auto-detected embedding cache: {abs_path}")
            return abs_path
            
    # 如果没找到，也返回绝对路径
    default_dir = os.path.join(data_root, "embeddings") if data_root else "./data/embeddings"
    os.makedirs(default_dir, exist_ok=True)
    return os.path.abspath(os.path.join(default_dir, filename))
def get_data_loader(cfg):
    task_name = cfg.data.task_name
    data_root = cfg.data.get("data_root", "./data")
    print(f"🚀 Loading task: {task_name}")

    train_inputs, train_y = [], []
    test_inputs, test_y = [], []

    # --- 1. 加载原始数据 ---
    if task_name in ['BindingDB_Ki', 'BindingDB_IC50', 'KIBA', 'DAVIS']:
        try:
            raw_data = load_DTI(
                name=task_name,
                split_method=cfg.data.get("split_method", "random"),
                max_smiles_length=cfg.data.get("max_smiles_length", None),
                max_protein_length=cfg.data.get("max_protein_length", None)
            )
            train_inputs_raw, train_y_raw, _, _, test_inputs_raw, test_y_raw = raw_data
            
            # --- [修复 1] 正确提取 Input ---
            if hasattr(train_inputs_raw, 'iloc'):
                 train_inputs = train_inputs_raw['Target'].tolist()
                 test_inputs = test_inputs_raw['Target'].tolist()
            else:
                 train_inputs = [x[1] for x in train_inputs_raw]
                 test_inputs = [x[1] for x in test_inputs_raw]

            # --- [修复 2] 正确提取 Label (解决 zip 短板问题) ---
            # 如果是 DataFrame (二维)，转为 flatten 的列表
            if hasattr(train_y_raw, 'iloc'):
                # 尝试取 'Y' 列，如果不行就取第 0 列
                try:
                    train_y = train_y_raw['Y'].values.flatten().tolist()
                    test_y = test_y_raw['Y'].values.flatten().tolist()
                except KeyError:
                    train_y = train_y_raw.iloc[:, 0].values.flatten().tolist()
                    test_y = test_y_raw.iloc[:, 0].values.flatten().tolist()
            else:
                train_y = train_y_raw
                test_y = test_y_raw

        except Exception as e:
            print(f"⚠️ Standard load failed ({e}), trying fallback to CSV...")
            # Fallback 逻辑保持简化
            raise e

    elif task_name == 'ESOL':
        train_ids, train_y, _, _, test_ids, test_y = load_ESOL()
        train_inputs = train_ids.flatten().tolist()
        test_inputs = test_ids.flatten().tolist()
    else:
        # 其他任务简单处理
        pass 

    print(f"📊 Raw Data Size: Train={len(train_inputs)}, Labels={len(train_y)}")
    
    # 确保长度一致，否则 zip 会丢数据
    assert len(train_inputs) == len(train_y), f"Mismatch! Inputs: {len(train_inputs)}, Labels: {len(train_y)}"

    # --- 2. 缓存过滤 (只保留命中缓存的数据) ---
    print("🧹 Filtering data by cache...")
    emb_folder_cfg = cfg.data.get("embedding_folder", None)
    cache_path = _find_embedding_cache(data_root, emb_folder_cfg)
    
    fm_model = esm_model(
        esm_model_name=cfg.data.get("esm_model_path", "facebook/esm2_t30_150M_UR50D"), 
        avoid_loading_model=False, 
        rep_cache_path=cache_path,
        use_cache=True
    )
    
    # --- [新增] 调试：打印前几个 Key 看看长什么样 ---
    print("\n🔍 --- DEBUG: Key Matching ---")
    cache_keys = list(fm_model.aaseq_rep_map.keys())
# ... (前面的代码保持不变，直到 print("🔍 --- DEBUG: Key Matching ---") 那里) ...

    # --- [升级版] 调试与过滤逻辑 ---
    print("\n🔍 --- DEBUG: Deep Inspection ---")
    
    # 1. 提取字典里的任意一个 Key 进行解剖
    cache_sample = list(fm_model.aaseq_rep_map.keys())[0] if fm_model.aaseq_rep_map else "EMPTY"
    input_sample = train_inputs[0] if train_inputs else "EMPTY"
    
    print(f"1. Cache Key Sample (Raw): '{cache_sample}'")
    print(f"   Length: {len(cache_sample)}")
    print(f"2. Input Data Sample (Raw): '{input_sample}'")
    print(f"   Length: {len(input_sample)}")
    
    # 3. 尝试暴力匹配测试
    print("3. Running heuristic check...")
    # 移除所有非字母字符 (空格、换行、制表符)
    import re
    clean_cache = re.sub(r'[^A-Z]', '', str(cache_sample).upper())
    clean_input = re.sub(r'[^A-Z]', '', str(input_sample).upper())
    
    print(f"   Cleaned Cache: '{clean_cache[:50]}...'")
    print(f"   Cleaned Input: '{clean_input[:50]}...'")
    
    if clean_cache == clean_input:
        print("   ✅ MATCH FOUND after cleaning! (Format mismatch detected)")
    else:
        print("   ❌ NO MATCH even after cleaning. (Datasets might be different)")
    print("---------------------------------\n")

    def normalize_seq(s):
        """强力清洗函数：转大写，去空格，去换行"""
        if not isinstance(s, str): s = str(s)
        # 很多生物序列文件会有换行符或空格，必须去掉
        return "".join(s.split()).upper()

    def filter_and_retrieve(inputs, labels, model, desc="Filtering"):
        valid_embs = []
        valid_texts = []
        valid_labels = []
        miss_count = 0
        
        # 为了加速，先把 Cache 的 Key 也全部清洗一遍并建立映射
        # 注意：这会消耗一些内存，但为了匹配是值得的
        print(f"[{desc}] Pre-processing cache keys for robust matching...")
        # 原始Key -> Embedding
        raw_cache = model.aaseq_rep_map
        # 清洗后Key -> 原始Key (用于取值)
        # 只有当两个不同的原始Key清洗后变成同一个时会有冲突，这里暂且忽略
        clean_cache_map = {normalize_seq(k): k for k in raw_cache.keys()}
        
        print(f"[{desc}] Start matching...")
        for seq, label in zip(tqdm(inputs, desc=desc), labels):
            # 清洗输入
            clean_seq = normalize_seq(seq)
            
            if clean_seq in clean_cache_map:
                # 命中！通过清洗后的Key找到原始Key，再取Embedding
                original_key = clean_cache_map[clean_seq]
                emb = raw_cache[original_key]
                
                # 格式转换
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu()
                elif isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                elif isinstance(emb, list):
                    emb = torch.tensor(emb)
                
                if emb.dim() == 2 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                    
                valid_embs.append(emb)
                valid_texts.append(original_key) # 存原始的还是清洗后的都可以
                valid_labels.append(label)
            else:
                miss_count += 1
                
        print(f"   👉 [{desc}] Kept: {len(valid_embs)} | Dropped: {miss_count}")
        
        if len(valid_embs) == 0:
            # 此时如果还是0，那就是真没数据了
            print(f"⚠️ WARNING: Zero matches for {desc}. This is critical.")
            # 为了防止程序直接崩掉无法看日志，我们这里如果不抛错，后续也会报错
            # 但既然你是做 case study，可能只要一部分数据就行
            # 如果训练集空了，必须报错
            if desc == "Train":
                raise ValueError(f"CRITICAL: No data found in cache for {desc}! Check 'Deep Inspection' logs above.")

        # 如果没有数据，返回空 tensor
        if len(valid_embs) == 0:
            return [], torch.tensor([]), np.array([])
            
        return valid_texts, torch.stack(valid_embs), np.array(valid_labels)

    train_inputs, train_emb, train_y = filter_and_retrieve(train_inputs, train_y, fm_model, "Train")
    test_inputs, test_emb, test_y = filter_and_retrieve(test_inputs, test_y, fm_model, "Test")

    return {
        "train": {"text": train_inputs, "emb": train_emb, "label": train_y},
        "test": {"text": test_inputs, "emb": test_emb, "label": test_y}
    }