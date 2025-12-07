### 核心设计哲学：**将“数据变换”与“模型推理”彻底分离**

原代码最大的问题是把 PCA、OT（最优传输）等数学逻辑写进了 `Dataset` 甚至 `Trainer` 里。我们的核心重构思路是：**所有的 DSP 操作（Spectral-ICRL）和降维操作（PCA），都只是“预处理（Preprocessing）”的一部分，模型根本不需要知道它读入的是 DCT 系数还是 PCA 特征，它只负责读 Tensor。**

-----

### 1\. 新版目录结构 (The Blueprint)

这是重构后的文件树，非常清爽：

```text
Spectral/
├── configs/                  # [控制中心] 所有的实验变量都在这里
│   ├── baselines/
│   │   ├── pca_esol.yaml     # Baseline: PCA
│   │   └── ot_esol.yaml      # Baseline: OT-PCA
│   └── ours/
│       ├── spectral_full.yaml      # Ours: 完整版
│       ├── ablation_no_reorder.yaml # Ablation: 去掉重排序
│       └── ablation_no_residual.yaml # Ablation: 去掉高频残差
│
├── src/
│   ├── data/                 # [数据层] 负责加载和预处理
│   │   ├── loader.py         # 加载原始 .pt/.npy 缓存
│   │   └── processors/       # [核心] 所有的数学变换都在这里！
│   │       ├── base.py       # 定义接口
│   │       ├── pca.py        # 封装 sklearn PCA
│   │       └── spectral.py   # [你的主场] Fiedler + DCT + Shaping
│   │
│   ├── model/                # [模型层] 极简封装
│   │   ├── llama_wrapper.py  # 专门适配 Llama-3.1-8B
│   │   └── prompt_builder.py # 专门负责拼字符串
│   │
│   └── utils/
│       └── metrics.py        # 统一的评估指标 (RMSE, Pearson)
│
├── run.py                    # [入口] 唯一的启动脚本
└── preprocess.py             # [工具] 离线预处理脚本（推荐）
```

-----

### 2\. 详细模块设计与实现思路

#### A. 配置层 (`configs/*.yaml`) —— 一切的指挥官

我们不再用 `argparse` 传几十个参数，而是用 YAML 文件控制实验流。

**示例：你的 Spectral-ICRL 实验配置**

```yaml
experiment_name: "spectral_toxicity_v1"
seed: 42

dataset:
  name: "Toxicity"
  path: "./data/toxicity_cache.pt"
  split_ratio: [0.8, 0.2]

# 核心差异点：方法配置
method:
  name: "spectral"  # 对应 SpectralProcessor
  params:
    reorder: true   # [消融实验开关] 是否做 Fiedler 重排序
    dct_k: 16       # 保留前16个低频
    residual: true  # [消融实验开关] 是否加高频残差
    align_stats:    # 谱整形的对齐目标 (Llama3 的统计值)
      mean: 0.0
      std: 0.02

llm:
  model_path: "meta-llama/Llama-3.1-8B-Instruct"
  shots: 20
  batch_size: 4
```

#### B. 数据处理层 (`src/data/processors/`) —— 你的代码主战场

这是实现你 idea 的核心区域。我们使用**策略模式（Strategy Pattern）**。

  * **`base.py`**: 定义一个标准接口 `transform(train_data, test_data) -> (new_train, new_test)`。
  * **`spectral.py` (你的实现)**:
    ```python
    class SpectralProcessor(BaseProcessor):
        def __init__(self, config):
            self.reorder = config.params.reorder
            self.residual = config.params.residual
            # ... 其他参数

        def fit_transform(self, train_emb, test_emb):
            # 1. 计算 Fiedler Vector (如果 reorder=True)
            if self.reorder:
                perm_idx = self.compute_fiedler(train_emb)
                train_emb = train_emb[:, perm_idx]
                test_emb = test_emb[:, perm_idx]
            
            # 2. DCT 变换 (scipy.fft)
            train_dct = dct(train_emb)
            test_dct = dct(test_emb)
            
            # 3. 截断与残差
            # ... 实现你的低频保留 + 高频能量计算逻辑 ...
            
            # 4. 谱整形 (Spectral Shaping)
            # ... 对齐均值方差 ...
            
            return train_final, test_final
    ```

**如何做消融实验？**
很简单，只需要在 Config 里把 `reorder: true` 改成 `false`，代码逻辑自动跳过第 1 步。这就实现了一个“不做排序直接 DCT”的消融实验。

#### C. 数据集层 (`src/data/dataset.py`) —— 极简主义

既然复杂的数学变换都在 Processor 里做完了，`UnifiedDataset` 就变得极其简单。它不需要知道什么是 PCA，什么是 DCT，它只认 Tensor。

```python
class UnifiedDataset(Dataset):
    def __init__(self, texts, embeddings, labels, prompt_builder):
        self.texts = texts
        self.embeddings = embeddings # 这里的 embedding 已经是处理好(如17维)的向量
        self.labels = labels
        self.builder = prompt_builder

    def __getitem__(self, idx):
        # 只需要负责把 embedding 塞进 Prompt 模板里
        input_ids, labels = self.builder.build(
            self.texts[idx], 
            self.embeddings[idx], 
            self.labels[idx]
        )
        return {"input_ids": input_ids, "labels": labels}
```

#### D. 模型层 (`src/model/prompt_builder.py`) —— 针对 Llama-3 的特化

既然你只用 Llama-3.1-8B，我们就把原来的垃圾代码（支持十几种模型的判断逻辑）全删掉。

```python
class Llama3PromptBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Llama 3 特有的 special tokens
        self.header = "<|begin_of_text|><|start_header_id|>system..." 
    
    def build(self, text, embedding, label):
        # 优雅地构建 Prompt，不再手动算 len()
        # 直接生成带有占位符的 input_ids
        pass
```

-----

### 3\. 这个架构如何满足你的四个要求？

1.  **支持各种数据集**

      * **实现：** `configs/dataset` 部分配置路径。只要你能把数据转成 `(N, D)` 的 Tensor 存下来，代码就能跑。无论是分子、蛋白还是音频，对于 `Processor` 来说都是矩阵。

2.  **自由替换 Baseline 做对比**

      * **实现：** 只需要切换 `config.yaml`。
          * 跑 PCA Baseline: 使用 `configs/baselines/pca.yaml` (其中 `method: pca`).
          * 跑 你的方法: 使用 `configs/ours/spectral.yaml` (其中 `method: spectral`).
      * 代码逻辑完全不用动，`main.py` 会根据 config 自动加载 `PCAProcessor` 或 `SpectralProcessor`。

3.  **实现你的 Idea (Spectral-ICRL)**

      * **实现：** 你只需要专注编写 `src/data/processors/spectral.py` 这一个文件。所有的 DSP 逻辑（重排序、DCT、残差）都封装在这个类里，不会污染其他代码。

4.  **实现消融实验**

      * **实现：** 如前所述，通过 Config 的布尔开关 (`true/false`) 控制 `SpectralProcessor` 内部的逻辑分支。你可以一键生成 `table_ablation_results.csv`。

### 4\. 总结与建议

这个架构的核心在于\*\*“预处理前置”\*\*。

  * 原代码的混乱在于它在训练循环里做数据处理。
  * 新架构建议你：**先用 `preprocess.py` 把所有数学变换做完，存成处理好的小 Tensor，然后再喂给 LLM。**

这样，你的训练/推理脚本会跑得飞快，而且逻辑极其清晰。你的大作业报告里可以直接贴 `spectral.py` 的核心代码片段，评委一看就知道你的工程能力和 DSP 功底都很扎实。