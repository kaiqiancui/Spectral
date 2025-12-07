import re
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

def extract_answer(text):
    """
    从 Llama-3 的生成文本中提取数值。
    支持格式: "5.32", "Answer: 5.32", "The value is 5.32"
    """
    # 1. 移除特殊字符
    text = text.replace("<|eot_id|>", "").strip()
    
    # 2. 尝试正则匹配浮点数 (优先匹配行尾的数字)
    # 匹配格式: 整数或小数, 允许负号
    matches = re.findall(r"-?\d+\.?\d*", text)
    
    if len(matches) > 0:
        # 通常最后一个数字是答案（因为我们让它最后输出）
        try:
            return float(matches[-1])
        except:
            return None
    return None

def calculate_metrics(preds, targets):
    """
    计算 RMSE, Pearson, Spearman
    """
    # 过滤掉无效预测 (None)
    valid_preds = []
    valid_targets = []
    
    for p, t in zip(preds, targets):
        if p is not None:
            valid_preds.append(p)
            valid_targets.append(t)
            
    if len(valid_preds) == 0:
        return {"rmse": 0.0, "pearson": 0.0, "valid_count": 0}
        
    preds_arr = np.array(valid_preds)
    targets_arr = np.array(valid_targets)
    
    rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))
    pearson, _ = pearsonr(targets_arr, preds_arr)
    spearman, _ = spearmanr(targets_arr, preds_arr)
    
    return {
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "valid_count": len(valid_preds),
        "total_count": len(targets)
    }