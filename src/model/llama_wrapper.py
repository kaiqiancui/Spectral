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

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        feature_embeds=None, 
        labels=None, 
        **kwargs
    ):
        """
        Args:
            input_ids: [Batch, Seq_Len] 文本输入的 Token IDs
            attention_mask: [Batch, Seq_Len] 文本的 Mask
            feature_embeds: [Batch, Feature_Dim] 原始的科学特征 (UniMol/ESM)
            labels: [Batch, Seq_Len] 标签，用于计算 Loss
        """
        
        # 1. 获取文本的 Embedding [Batch, Seq_Len, Hidden_Dim]
        text_embeds = self.get_input_embeddings()(input_ids)
        
        # 2. 处理科学特征
        if feature_embeds is not None:
            # 确保类型匹配
            feature_embeds = feature_embeds.to(text_embeds.dtype).to(text_embeds.device)
            
            # 投影: [Batch, Feature_Dim] -> [Batch, LLM_Dim]
            # 如果 feature_embeds 是 [Batch, Dim]，我们需要 unsqueeze 成 [Batch, 1, Dim]
            if len(feature_embeds.shape) == 2:
                feature_embeds = feature_embeds.unsqueeze(1)
            
            projected_embeds = self.projector(feature_embeds) # [Batch, 1, LLM_Dim]
            
            # 3. 拼接 Embedding: [Science, Text]
            inputs_embeds = torch.cat([projected_embeds, text_embeds], dim=1)
            
            # 4. 扩展 Attention Mask
            # 文本的 Mask 是 [1, 1, 0, ...]，我们需要在前面加一个 1 给科学特征
            if attention_mask is not None:
                # 创建一个全 1 的列: [Batch, 1]
                feature_mask = torch.ones(
                    (attention_mask.shape[0], projected_embeds.shape[1]), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                attention_mask = torch.cat([feature_mask, attention_mask], dim=1)
            
            # 5. 扩展 Labels (如果存在)
            # 科学特征部分不应该计算 Loss，所以 Label 填充 -100 (Ignore Index)
            if labels is not None:
                feature_labels = torch.full(
                    (labels.shape[0], projected_embeds.shape[1]), 
                    -100, 
                    dtype=labels.dtype, 
                    device=labels.device
                )
                labels = torch.cat([feature_labels, labels], dim=1)
                
        else:
            inputs_embeds = text_embeds

        # 6. 调用 LLM
        # 注意：这里我们传入 inputs_embeds，而不是 input_ids
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
        return outputs

    # 为了兼容 HuggingFace Trainer 的 save_pretrained
    def save_pretrained(self, save_directory):
        self.llm.save_pretrained(save_directory)
        # 还需要单独保存 projector，或者将其作为模块保存
        torch.save(self.projector.state_dict(), f"{save_directory}/projector.pt")