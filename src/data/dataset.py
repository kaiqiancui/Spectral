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
        # 1. 采样 Shots (已实现分层采样，好评)
        shots = self.sample_shots(self.n_shots)
        
        # 2. 构建 Prompt (需严格对齐原仓库 MOL 模板)
        # 原仓库逻辑: System Prompt -> (Shot Text + Feature + Label) * N -> Test Text + Feature -> Answer:
        
        full_text = self.system_prompt + "\n\n"
        feature_stack = []

        # 定义 Marker (原仓库 README 提到后来改用了 "(" 和 ")" 或者 "[REP]")
        rep_start = "(" 
        rep_end = ")"

        # 拼接 Shots
        for shot in shots:
            # 严格复刻原仓库格式
            full_text += f"Drug SMILES: < {shot['input_text']} >\n"
            full_text += f"Given the SMILES sequence of the drug molecule, answer the following question.\n"
            full_text += f"Question: What is the {self.config.property_name} of the drug molecule?\n"
            full_text += "Use this molecular representation to answer the question: "
            
            # 注入点
            full_text += f"{rep_start} <|feature|> {rep_end}\n" 
            
            # Label
            full_text += f"Answer: {shot['label']}\n\n"
            
            feature_stack.append(shot['feature_emb'])

        # 拼接 Test Query
        full_text += f"Drug SMILES: < {test_sample['input_text']} >\n"
        full_text += f"Given the SMILES sequence of the drug molecule, answer the following question.\n"
        full_text += f"Question: What is the {self.config.property_name} of the drug molecule?\n"
        full_text += "Use this molecular representation to answer the question: "
        
        full_text += f"{rep_start} <|feature|> {rep_end}\n"
        full_text += "Answer:" # 留白
        
        feature_stack.append(test_sample['feature_emb'])

        return full_text, torch.stack(feature_stack)
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
    def sample_shots(self, n_shots):
        """
        复刻原仓库 utils.py -> stratified_sample 逻辑
        """
        if self.config.sampling_strategy == "stratified":
            # 假设 memory_bank 已经按 label 排好序了
            total_candidates = len(self.memory_bank['labels'])
            
            # 计算间隔
            indices = np.linspace(0, total_candidates - 1, n_shots, dtype=int)
            
            shots = []
            for idx in indices:
                shots.append({
                    'feature_emb': self.memory_bank['features'][idx],
                    'label': self.memory_bank['labels'][idx],
                    'input_text': self.memory_bank['text'][idx]
                })
            return shots
        else:
            # 随机采样
            # ...