import torch
import os
from src.config import load_config
from src.data.loader import get_data_loader
from src.data.processors.spectral import SpectralProcessor
# from src.data.processors.pca import PCAProcessor # 未来实现

def main():
    # 1. 加载配置
    cfg = load_config()
    print(f"🚀 Starting Preprocessing for {cfg.experiment.name}")
    print(f"   Method: {cfg.method.name}")
    print(f"   Target Dim: {cfg.method.params.target_dim}")

    # 2. 加载数据 (Raw Text + Raw Embeddings)
    raw_data = get_data_loader(cfg)
    train_data = raw_data['train']
    test_data = raw_data['test']

    # 3. 初始化处理器 (Strategy Pattern)
    if cfg.method.name == 'spectral':
        processor = SpectralProcessor(cfg)
    elif cfg.method.name == 'pca':
        # processor = PCAProcessor(cfg)
        raise NotImplementedError("PCA processor not implemented yet")
    else:
        raise ValueError(f"Unknown method: {cfg.method.name}")

    # 4. 执行变换 (核心步骤)
    print("🔄 Running Processor...")
    train_emb_processed, test_emb_processed = processor.fit_transform(
        train_data['emb'], 
        test_data['emb']
    )

    # 5. 保存结果
    save_path = os.path.join(cfg.experiment.save_dir, "processed_data.pt")
    print(f"💾 Saving processed data to {save_path}...")
    
    torch.save({
        "train": {
            "text": train_data['text'],
            "emb": train_emb_processed, # 这是处理后的 (N, 16)
            "label": train_data['label']
        },
        "test": {
            "text": test_data['text'],
            "emb": test_emb_processed,
            "label": test_data['label']
        },
        "config": cfg
    }, save_path)
    
    print("✅ Preprocessing Done!")

if __name__ == "__main__":
    main()