import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict
class ICRLDataset(Dataset):
    def __init__(self, data_split, full_train_data, tokenizer, config):
        self.data = data_split
        self.memory_bank = full_train_data # 用于采样 shots
        self.tokenizer = tokenizer
        self.n_shots = config.n_shots
        
        # 加载原仓库的 Prompt 文件
        with open(config.prompt_path, 'r') as f:
            self.system_prompt = f.read() 
            # 原 Prompt 里可能包含 "[REP]" 字符串说明，这没关系，作为文本保留
            # 我们实际注入特征的位置使用 <|feature|> token

    def construct_icl_sample(self, test_sample):
        # 1. 采样 Shots
        shots = self.sample_shots(self.n_shots)
        
        full_text = self.system_prompt + "\n\n"
        feature_stack = []

        # 2. 拼接 Shots (Interleaved)
        # 格式参考原仓库: "Protein: <seq> [REP] \n Stability: <label>"
        for shot in shots:
            full_text += f"Protein: {shot['input_text']}\n"
            full_text += "Representation: <|feature|>\n" # 占位符
            full_text += f"Stability: {shot['label']}\n\n"
            feature_stack.append(shot['feature_emb'])

        # 3. 拼接 Test Query
        full_text += f"Protein: {test_sample['input_text']}\n"
        full_text += "Representation: <|feature|>\n" # 占位符
        full_text += "Stability:" # 留白等待生成
        feature_stack.append(test_sample['feature_emb'])

        return full_text, torch.stack(feature_stack) # [N_shots+1, Feature_Dim]

    def __getitem__(self, index):
        item = self.data[index]
        text, features = self.construct_icl_sample(item)
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        
        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "feature_embeds": features # 这里的 features 包含了 shots 和 test 的所有特征
        }