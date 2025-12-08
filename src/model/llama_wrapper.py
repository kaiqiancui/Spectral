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
        # 1. 获取文本 Embedding [Batch, Seq_Len, Hidden_Dim]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 2. 投影特征 [Batch, N_features, Feature_Dim] -> [Batch, N_features, Hidden_Dim]
        # 注意：这里的 feature_embeds 包含了一个样本中所有的 shot 特征 + test 特征
        projected_features = self.projector(feature_embeds)

        # 3. 核心：替换逻辑 (Replacement)
        # 找到 input_ids 中所有 <|feature|> 的位置
        feature_mask = (input_ids == self.feature_token_id)
        
        # 校验：确保文本里的占位符数量 和 传入的特征数量一致 (Debug用)
        # batch_size = input_ids.shape[0]
        # assert feature_mask.sum() == projected_features.shape[0] * projected_features.shape[1]

        # 执行替换
        # inputs_embeds[mask] 会返回对应位置的向量，我们需要把 projected_features 填进去
        # 注意：projected_features 需要展平以匹配 mask 的非零元素顺序
        # [Batch, N_features, Hidden] -> [Batch * N_features, Hidden] (如果 batch>1 需要小心顺序)
        
        # 更加鲁棒的写法：
        if feature_mask.any():
            # 确保 projected_features 的总数匹配 mask 的总数
            target_dtype = inputs_embeds.dtype
            
            # 将 projected_features 展平为 [Total_Features_In_Batch, Hidden_Dim]
            flat_features = projected_features.view(-1, projected_features.shape[-1]).to(target_dtype)
            
            # 这里的赋值是按内存顺序进行的，只要 Dataset 构建时顺序一致即可
            inputs_embeds[feature_mask] = flat_features

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