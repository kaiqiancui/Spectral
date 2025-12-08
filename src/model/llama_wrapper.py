import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.components import build_projector

class LlamaWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 加载基础 LLM
        print(f"Loading Llama model from {config.model_name_or_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.get("use_bf16", True) else torch.float16,
            device_map="auto"
        )
        
        # 2. 初始化 Projector
        # 注意：Projector 需要放到与 LLM 相同的设备上，或者在 forward 时处理
        self.projector = build_projector(config)
        self.projector.to(self.llm.device).to(self.llm.dtype)

        # 冻结 LLM 参数 (如果只需要训练 Projector)
        if config.get("freeze_llm", True):
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
            # Projector 必须可训练
            for param in self.projector.parameters():
                param.requires_grad = True
        special_tokens = {"additional_special_tokens": ["<|feature|>"]}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        feature_token_id = tokenizer.convert_tokens_to_ids("<|feature|>")
        

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(self, input_ids, feature_embeds, attention_mask=None, **kwargs):
        # 1. 基础 Embedding
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 2. 投影特征
        # feature_embeds shape: [Batch, N_shots+1, Science_Dim]
        projected_features = self.projector(feature_embeds) # -> [Batch, N_shots+1, LLM_Dim]
        
        # 3. 核心修正：物理拼接注入 (复刻原仓库逻辑)
        # 假设 input_ids 中包含特殊的 placeholder token (例如 id=128256 对应 <|feature|>)
        # 我们需要把 inputs_embeds 在这个位置切开，塞入 projected_features
        
        batch_size = inputs_embeds.shape[0]
        new_embeds_list = []
        new_attn_mask_list = [] # 如果你需要处理 attention mask
        
        # 原仓库逻辑简化版：
        for b in range(batch_size):
            # 找到占位符的位置
            placeholder_mask = (input_ids[b] == self.feature_token_id)
            # 获取非占位符的文本 embedding 片段
            # 这里需要非常精细的操作：split -> cat (feature) -> split -> cat
            # 为了简化实现且保证稳健，建议在 Dataset 层就不要把 feature 变成 token id
            # 而是直接传 list of embeddings 和 list of text parts。
            
            # 但为了兼容目前的架构，我们使用替换法，但必须确保 projected_features 
            # 的形状 [Batch, N, Dim] 能被正确 reshape 进 inputs_embeds
            
            if placeholder_mask.sum() != projected_features.shape[1]:
                 raise ValueError(f"Prompt 中的占位符数量 ({placeholder_mask.sum()}) 与 特征数量 ({projected_features.shape[1]}) 不匹配！")
            
            # 使用 Mask 赋值 (前提：Projector 输出是 [Batch, N_features, LLM_Dim])
            # 并且 input_ids 里每个特征只占 1 个 token
            inputs_embeds[b][placeholder_mask] = projected_features[b].to(inputs_embeds.dtype)

        # 4. 调用 LLM
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
    # 为了兼容 HuggingFace Trainer 的 save_pretrained
    def save_pretrained(self, save_directory):
        self.llm.save_pretrained(save_directory)
        # 还需要单独保存 projector，或者将其作为模块保存
        torch.save(self.projector.state_dict(), f"{save_directory}/projector.pt")