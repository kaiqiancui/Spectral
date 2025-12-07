from .utils.utils import load_ESOL, load_DTI, load_Fluorescence

class TaskFactory:
    @staticmethod
    def load_raw_data(task_name, config):
        print(f"📖 Loading Task: {task_name}")
        
        # --- 1. 分子任务 (ESOL) ---
        if task_name == 'ESOL':
            # load_ESOL 返回: (train_x, train_y, val_x, val_y, test_x, test_y)
            # 我们只需要 train 和 test
            data = load_ESOL()
            train_x, train_y = data[0], data[1]
            test_x, test_y = data[4], data[5]
            
            # 标准化封装：单模态统一用 input1
            return {
                'train': [{'input1': x, 'label': y} for x, y in zip(train_x, train_y)],
                'test':  [{'input1': x, 'label': y} for x, y in zip(test_x, test_y)]
            }

        # --- 2. DTI 任务 (双模态) ---
        elif task_name in ['BindingDB_Ki', 'DAVIS']:
            # load_DTI 返回 DataFrame
            train_in, train_y, _, _, test_in, test_y = load_DTI(name=task_name)
            
            # 标准化封装：双模态用 input1 (Drug) 和 input2 (Target)
            # 注意：需根据原代码确认 train_in 是 DataFrame 还是 List
            # 假设是 DataFrame，列名为 Drug, Target
            train_data = []
            for i in range(len(train_in)):
                train_data.append({
                    'input1': train_in.iloc[i]['Drug'], 
                    'input2': train_in.iloc[i]['Target'],
                    'label': train_y.iloc[i]['Y']
                })
                
            test_data = []
            for i in range(len(test_in)):
                test_data.append({
                    'input1': test_in.iloc[i]['Drug'], 
                    'input2': test_in.iloc[i]['Target'],
                    'label': test_y.iloc[i]['Y']
                })
                
            return {'train': train_data, 'test': test_data}

        else:
            raise ValueError(f"Unknown task: {task_name}")