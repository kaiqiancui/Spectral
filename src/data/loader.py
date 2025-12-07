# src/data/loader.py
from .tasks import TaskLoader
from .cache import EmbeddingCache
from .dataset import UnifiedICLDataset

def get_data_loader(cfg):
    # 1. 加载原始文本 (复用原逻辑)
    train_txt, train_y, test_txt, test_y = TaskLoader.load(cfg.data.task_name, cfg)
    
    # 2. 加载缓存 (独立模块)
    cache = EmbeddingCache(cfg.data.cache_path)
    
    # 3. 查表并对齐 (Intersection)
    print("⚙️ Aligning Data with Cache...")
    train_embs, train_mask = cache.get_batch(train_txt)
    test_embs, test_mask = cache.get_batch(test_txt)
    
    # 4. 过滤 Label
    train_y_filtered = train_y[train_mask]
    test_y_filtered = test_y[test_mask]
    train_txt_filtered = [train_txt[i] for i in train_mask]
    test_txt_filtered = [test_txt[i] for i in test_mask]

    # 5. 返回 Dataset
    return {
        "train": UnifiedICLDataset(train_txt_filtered, train_embs, train_y_filtered, cfg),
        "test": UnifiedICLDataset(test_txt_filtered, test_embs, test_y_filtered, cfg)
    }