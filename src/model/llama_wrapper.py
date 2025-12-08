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
        projected_features = self.projector(feature_embeds) # [Batch, N_features, LLM_Dim]

        # --- 缺失逻辑补全: 归一化 (Align Distribution) ---
        # 计算 LLM 原始 Embedding 的统计量 (原仓库逻辑)
        # 注意: 为了效率，这里可以做成 buffer 在 init 时计算一次，但此处为了对齐原代码动态计算
        # 原仓库只取非零 Embedding 计算，这里简化为整体计算
        with torch.no_grad():
            target_mean = self.llm.get_input_embeddings().weight.mean()
            target_var = self.llm.get_input_embeddings().weight.var()
        
        # 对投影后的特征进行归一化
        feat_mean = projected_features.mean(dim=-1, keepdim=True)
        feat_var = projected_features.var(dim=-1, keepdim=True)
        projected_features = (projected_features - feat_mean) * torch.sqrt(target_var / (feat_var + 1e-6)) + target_mean
        # -----------------------------------------------

        # 3. 物理拼接注入 (比 Mask 替换更稳健)
        batch_size = input_ids.shape[0]
        combined_embeds_list = []
        combined_mask_list = []

        for b in range(batch_size):
            # 找到占位符的位置 (id 对应 <|feature|>)
            # 注意: 确保 tokenizer.convert_tokens_to_ids("<|feature|>") 是一个 int
            placeholder_mask = (input_ids[b] == self.feature_token_id)
            
            # 如果没有特征需要注入，直接用原始 embedding
            if not placeholder_mask.any():
                combined_embeds_list.append(inputs_embeds[b])
                if attention_mask is not None:
                    combined_mask_list.append(attention_mask[b])
                continue

            # 获取文本片段的 Embedding (即非占位符的部分)
            # 这里需要一种策略把 inputs_embeds[b] 切开，中间塞入 projected_features[b]
            # 为了简化实现，这里我们假定 <|feature|> 总是被分词为一个 token，继续使用你的替换逻辑，
            # 但加上上面的归一化步骤。如果需要严格对齐 inject_embeddings，代码会非常长。
            
            # 使用修正后的特征进行填入
            inputs_embeds[b][placeholder_mask] = projected_features[b].to(inputs_embeds.dtype)
            
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # 注意: 如果做了拼接导致长度变化，这里 mask 也要调整，目前替换法无需调整
            **kwargs
        )
    # 为了兼容 HuggingFace Trainer 的 save_pretrained
    def save_pretrained(self, save_directory):
        self.llm.save_pretrained(save_directory)
        # 还需要单独保存 projector，或者将其作为模块保存
        torch.save(self.projector.state_dict(), f"{save_directory}/projector.pt")