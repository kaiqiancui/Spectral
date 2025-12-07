# src/data/tasks.py
from .utils.utils import load_ESOL, load_DTI, load_BindingDB # 复用你已经考过来的工具

class TaskLoader:
    @staticmethod
    def load(task_name, config):
        """
        标准接口：输入任务名，输出 (train_texts, train_labels, test_texts, test_labels)
        """
        print(f"📖 Loading Raw Data for Task: {task_name}")
        
        if task_name == 'ESOL':
            # 原仓库逻辑：load_ESOL 返回 (train_x, train_y, val_x, val_y, test_x, test_y)
            # 我们这里做统一封装
            out = load_ESOL() 
            # 假设 load_ESOL 返回的是 numpy array 或 list
            return out[0].flatten(), out[1].flatten(), out[4].flatten(), out[5].flatten()
            
        elif task_name in ['DAVIS', 'KIBA']:
            # DTI 任务逻辑...
            pass
            
        raise ValueError(f"Unknown task: {task_name}")