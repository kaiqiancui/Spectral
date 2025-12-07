import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaWrapper(nn.Module):
    def __init__(self, config, projector):
        super().__init__()
        self.cfg = config
        model_path = config.llm.model_path
        
        print(f"🤖 Loading Llama-3.1 (Inference Mode)...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=config.experiment.device,
            load_in_4bit=config.llm.load_in_4bit,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # 添加占位符 <REP>
        self.rep_token = "<REP>"
        if self.rep_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.rep_token], special_tokens=True)
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.rep_token_id = self.tokenizer.convert_tokens_to_ids(self.rep_token)
        
        # 随机投影层 (不参与梯度更新)
        self.projector = projector
        for p in self.projector.parameters():
            p.requires_grad = False

        # 系统提示词
        prompt_path = config.llm.get("prompt_file", None)
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                self.system_message = f.read().strip()
        else:
            self.system_message = "Predict the value based on the representation."

    def apply_alignment(self, embs, align_stats):
        """
        核心逻辑：统计对齐 (Alignment)
        公式: (x - mu_x) / std_x * std_tgt + mu_tgt
        """
        if align_stats is None:
            return embs
            
        mu_x = embs.mean(dim=0, keepdim=True)
        std_x = embs.std(dim=0, keepdim=True) + 1e-8
        
        # 目标分布 (LLM 的 Embedding 分布)
        mu_tgt = align_stats['target_mean'].to(embs.device)
        std_tgt = align_stats['target_std'].to(embs.device)
        
        return (embs - mu_x) / std_x * std_tgt + mu_tgt

    def _build_prompt(self, text, label=None, is_shot=False):
        # 读取 Config 中的模板，而不是硬编码
        # 假设模板是: "The molecule is <REP>. Property is:"
        template = self.cfg.llm.prompt_template 
        
        # 替换 <REP> 占位符
        prompt = template.replace("<REP>", self.rep_token)
        prompt = prompt.replace("<SMILES>", text) # 如果模板里有原文占位符
        
        if is_shot:
            prompt += f" {label:.3f}\n" # Shot 结尾加 Label
        else:
            prompt += "" # Query 结尾留空让 LLM 续写
            
        return prompt

    @torch.inference_mode()
    def generate(self, net_input, align_stats=None):
        """
        Training-Free 推理函数
        align_stats: 从训练集计算出的对齐参数
        """
        query_texts = net_input['query_text']
        query_embs = net_input['query_emb']
        shots_batch = net_input['shots']
        
        batch_size = len(query_texts)
        full_prompts = []
        all_reps_list = []
        
        # 1. 拼接 Prompt 和 收集 Embeddings
        sys_header = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_message}<|eot_id|>"
        
        for i in range(batch_size):
            curr_prompt = sys_header
            curr_reps = []
            
            # Shots
            for shot in shots_batch[i]:
                curr_prompt += self._build_prompt(shot['text'], shot['label'], is_shot=True)
                curr_reps.append(shot['emb'])
            
            # Query
            curr_prompt += self._build_prompt(query_texts[i], label=None, is_shot=False)
            curr_reps.append(query_embs[i])
            
            full_prompts.append(curr_prompt)
            all_reps_list.append(torch.stack(curr_reps))
            
        # 2. Tokenize
        inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.llm.device)
        
        # 3. 投影 + 对齐 (Projection + Alignment)
        flat_reps = torch.cat(all_reps_list, dim=0).to(self.llm.device).to(self.llm.dtype)
        
        # Step A: 随机投影 (640 -> 4096)
        projected_reps = self.projector(flat_reps)
        
        # Step B: 统计对齐 (关键步骤!)
        # 让投影后的向量分布 看起来像 Llama 的 Token Embedding
        aligned_reps = self.apply_alignment(projected_reps, align_stats)
        
        # 4. 替换 Embedding
        inputs_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
        is_rep_token = (inputs.input_ids == self.rep_token_id)
        
        if is_rep_token.sum() == aligned_reps.shape[0]:
            inputs_embeds[is_rep_token] = aligned_reps
        else:
            # 截断保护
            min_len = min(is_rep_token.sum(), aligned_reps.shape[0])
            inputs_embeds[is_rep_token] = aligned_reps[:min_len]

        # 5. Generate (直接生成文本)
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False, # 确定性生成
            temperature=None
        )
        
        # 解码生成的文本
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_output