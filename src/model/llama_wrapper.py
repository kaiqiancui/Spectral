import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        model_path = config.llm.model_path
        
        print(f"🤖 Loading LLM: {model_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=config.experiment.device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # 1. 定义占位符 Token (用于定位 Embedding 插入点)
        # 我们使用 <REP> 作为通用占位符
        self.rep_token = "<REP>"
        if self.rep_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.rep_token], special_tokens=True)
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.rep_token_id = self.tokenizer.convert_tokens_to_ids(self.rep_token)
        
        # 2. 加载 Prompt 模板
        # 示例 Template: "Question: What is the property of <REP>? Answer:"
        self.prompt_template = config.llm.get("prompt_template", "Input: <REP>\nOutput:")

    def _build_prompt_text(self, text_data):
        """
        根据 Config 的模板构建文本 Prompt
        text_data: 包含 'input1', 'input2' 等原始文本的字典
        """
        prompt = self.prompt_template
        
        # 简单替换：如果模板里有 <INPUT> 之类的标签，可以用 text_data 替换
        # 这里假设模板主要是为了安放 <REP>
        # 如果是 DTI 任务，模板可能是 "Drug: <REP> Target: <REP> ..."
        # LlamaWrapper 不需要知道是 Drug 还是 Target，它只负责看到一个 <REP> 就准备填一个向量
        return prompt

    def forward(self, batch):
        # 仅用于训练或调试，通常我们用 generate
        pass

    @torch.inference_mode()
    def generate(self, batch):
        """
        执行推理
        batch: DataLoader yield 出的字典
        """
        device = self.llm.device
        batch_size = len(batch['input1']) # 假设 batch 包含 input1, input1_emb 等
        
        # 1. 构建纯文本 Prompt List
        prompts = [self._build_prompt_text(None) for _ in range(batch_size)]
        
        # 2. Tokenize
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 3. 获取 LLM 原始 Embedding
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 4. [核心] 替换 Embedding
        # 找到 input_ids 中等于 <REP> 的位置，替换为 batch 中的 embedding
        # 注意：这里假设 batch['input1_emb'] 已经是 [Batch, Hidden_Dim] (即投影后的)
        
        # 简单实现：假设每个 Prompt 只有一个 <REP>，且我们用 input1_emb 替换
        # 如果是双模态，需要更复杂的逻辑 (按顺序替换)
        
        rep_mask = (input_ids == self.rep_token_id)
        
        # 检查 batch 中有哪些 embedding
        # 我们的 Loader 产生了 input1_emb, input2_emb ...
        embeddings_to_insert = []
        if 'input1_emb' in batch:
            embeddings_to_insert.append(batch['input1_emb'].to(device).to(inputs_embeds.dtype))
        if 'input2_emb' in batch:
            embeddings_to_insert.append(batch['input2_emb'].to(device).to(inputs_embeds.dtype))
            
        # 这里的逻辑是将所有 embedding 拼起来还是分别替换？
        # 为了兼容原代码逻辑，通常是一个样本对应一个向量序列。
        # 如果是单模态，insert_emb 就是 [Batch, 1, 4096]
        
        if len(embeddings_to_insert) > 0:
            # 拼接多模态 (如果需要) 或者只取第一个
            # 简化起见：我们假设 batch['input1_emb'] 是主要的
            insert_emb = embeddings_to_insert[0] 
            
            # 确保维度匹配: [Batch, Seq_Len, Hidden]
            if insert_emb.dim() == 2:
                insert_emb = insert_emb.unsqueeze(1) # [Batch, 1, 4096]
            
            # 执行替换 (Scatter)
            # 注意：这要求 <REP> 的数量和 insert_emb 的序列长度一致
            # 这里做一个简化的假设：每个样本只替换一个位置
            inputs_embeds[rep_mask] = insert_emb.squeeze(1)

        # 5. Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        
        # 6. Decode
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded