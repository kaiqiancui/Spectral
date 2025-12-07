from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseProcessor(ABC):
    def __init__(self, config):
        self.cfg = config
        
    @abstractmethod
    def fit_transform(self, train_data, test_data):
        """
        核心接口：
        输入: (N, D) 的原始数据 (numpy array 或 torch tensor)
        输出: (N, K) 的处理后数据 (numpy array 或 torch tensor)
        注意: 必须只利用 train_data 计算统计量 (fit)，然后应用到 test_data (transform)
        """
        pass