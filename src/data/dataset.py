import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict

class SpectralDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, config):
        """
        Args:
            data: 处理后的数据列表 (来自 Processor)
            tokenizer: Llama Tokenizer
            config: 包含 prompt_file 路径的配置
        """
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.get("max_length", 512)
        
        # --- Prompt 加载逻辑 ---
        # 优先从配置读取 Prompt 文件路径，如果没有则使用默认格式
        prompt_path = config.get("prompt_file", None)
        if prompt_path and os.path.exists(prompt_path):
            print(f"Loading prompt template from {prompt_path}")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read().strip()
        else:
            print("No prompt file found, using default format.")
            self.prompt_template = "Question: {input}\nAnswer:" # 默认 fallback

    def __len__(self):
        return len(self.data)

    def apply_prompt(self, item):
        """
        将数据填入原仓库风格的 Prompt 模板。
        原仓库逻辑通常是简单的 Python f-string 格式化。
        """
        input_text = item.get("raw_text", "") # Processor 已经构建了一部分 (ICL Context)
        
        # 如果 Processor 只是传回了 input/output 字典，我们需要在这里格式化
        # 但之前的 SpectralProcessor 代码其实已经把 ICL Context 拼好了放在 raw_text 里
        # 所以这里我们主要负责加上 System Prompt 或者 Task Instruction
        
        # 假设 self.prompt_template 类似于:
        # "You are an expert... \n\n{context}\n\nQuestion: {input}\nAnswer:"
        
        # 这里做一个简单的处理：
        # 如果 prompt_template 里包含 {input}，我们就替换它
        # 否则，我们把 template 放在最前面作为 Instruction
        
        full_text = input_text
        
        # 如果 Processor 还没有完全格式化，我们可以在这里做
        # 但为了配合上一轮的代码，我们假设 raw_text 已经是 "Ctx... Q:.. A:" 的形式了
        # 唯一的区别是加上 Task Description (Prompt)
        
        if self.prompt_template:
             # 将 Prompt 放在最前面 (System Message / Instruction)
             full_text = f"{self.prompt_template}\n\n{input_text}"
             
        return full_text

    def __getitem__(self, index):
        item = self.data[index]
        
        # 1. 构建完整文本
        # 注意：这里的 full_text 仅包含文本部分，feature 还是通过 Tensor 传入
        text_input = self.apply_prompt(item)
        target_text = item['target_text']
        
        # 2. Tokenize
        # Llama 3.1 建议使用 <|begin_of_text|>，Tokenizer 通常会自动处理
        # 但如果是纯补全任务，可能需要手动加 EOS
        
        # 这里的逻辑是：Input + Target 一起编码，计算 Loss 时 Mask 掉 Input 部分
        full_sequence = text_input + " " + str(target_text) + self.tokenizer.eos_token
        
        tokenized = self.tokenizer(
            full_sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        # 3. 构造 Labels (Mask 掉 Question 部分，只预测 Answer)
        labels = input_ids.clone()
        
        # 找到 Answer 开始的位置 (简单的 heuristic)
        # 注意：这种 split 方法在某些 tokenizer 下不准，更严谨的方法是分开 tokenize
        # 这里为了简化，我们先假设 text_input 的长度就是 input 的长度
        input_len = len(self.tokenizer(text_input, truncation=True, max_length=self.max_length)["input_ids"])
        
        # Mask 掉 input 部分
        labels[:input_len] = -100
        # Mask 掉 Padding 部分
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # 这里的 Key 必须和 LlamaWrapper forward 中的参数名一致
            "feature_embeds": item["feature_embeds"] 
        }