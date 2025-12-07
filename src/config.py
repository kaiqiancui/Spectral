import os
import sys
from omegaconf import OmegaConf

def load_config(config_path=None, cli_args=None):
    """
    加载配置策略：
    1. 先加载默认配置 (configs/base.yaml)
    2. 如果指定了实验配置 (config_path)，覆盖默认配置
    3. 如果命令行传入了参数 (cli_args)，最后覆盖
    """
    # 1. 定位 base.yaml
    # 假设 src/config.py 在 src/ 目录下，向上两级找到 configs/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    base_config_path = os.path.join(project_root, "configs", "base.yaml")
    
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found at {base_config_path}")
        
    # 加载 Base
    cfg = OmegaConf.load(base_config_path)
    
    # 2. 加载特定实验配置 (Merge)
    if config_path:
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Experiment config not found at {config_path}")
        exp_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, exp_cfg)
        
    # 3. 命令行参数覆盖 (可选，方便快速调试)
    # 例如：python run.py method.params.reorder=false
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        
    # 4. 自动设置一些衍生路径 (避免代码里到处拼路径)
    # 例如：./logs/BindingDB_Ki/spectral_reorder_true
    method_suffix = f"{cfg.method.name}"
    if cfg.method.name == 'spectral':
        if cfg.method.params.reorder:
            method_suffix += "_reorder"
        else:
            method_suffix += "_no_reorder"
            
    cfg.experiment.save_dir = os.path.join(
        cfg.experiment.output_dir,
        cfg.data.task_name,
        method_suffix,
        cfg.experiment.name
    )
    
    # 创建输出目录
    os.makedirs(cfg.experiment.save_dir, exist_ok=True)
    
    return cfg

# 测试代码
if __name__ == "__main__":
    # 在这里简单测试一下能不能读到
    try:
        cfg = load_config()
        print("✅ Configuration loaded successfully!")
        print(OmegaConf.to_yaml(cfg))
    except Exception as e:
        print(f"❌ Error loading config: {e}")