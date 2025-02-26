import random
import os
import argparse
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs

def generate_hard_edf_instance(n):
    """
    生成一個 EDF 排程問題實例
    Args:
        n: 作業數量
    Returns:
        jobs: 每個作業以 (處理時間, 截止時間) 表示
    """
    jobs = []
    current_time = 0
    # 產生嚴格遞增的截止時間序列
    for i in range(n):
        p = random.randint(1, 10)  # 處理時間範圍可調整
        current_time += p
        d = current_time           # 截止時間等於累積處理時間
        jobs.append((p, d))
    
    # 加入干擾項，隨機微調約1/3的作業截止時間
    for _ in range(n // 3):
        idx = random.randint(0, n - 1)
        jobs[idx] = (jobs[idx][0], jobs[idx][1] + random.randint(0, 2))
    
    # 打亂順序
    random.shuffle(jobs)
    return jobs

def compute_edf_schedule(jobs):
    """
    根據 EDF 原則計算可行排程
    若累積處理時間超過某作業的截止時間，則視為不可行
    Args:
        jobs: 作業列表，每個作業為 (p, d)
    Returns:
        排程順序（以原始索引表示），或 "No feasible schedule"
    """
    # 取得原始索引
    indexed_jobs = list(enumerate(jobs))
    # 根據截止時間排序
    sorted_jobs = sorted(indexed_jobs, key=lambda x: x[1][1])
    current_time = 0
    schedule = []
    for idx, (p, d) in sorted_jobs:
        current_time += p
        if current_time > d:
            return "No feasible schedule"
        schedule.append(idx)
    return schedule

def generate_scheduling_prompt(jobs, template_type):
    """
    產生排程問題的 prompt，支援 base 和 qwen-instruct 格式
    """
    jobs_description = [(i, p, d) for i, (p, d) in enumerate(jobs)]
    if template_type == 'base':
        """This works for any base model"""
        prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the given jobs {jobs_description} (each as (job_id, processing_time, deadline)), create a feasible schedule that completes all jobs before their deadlines using Earliest Deadline First (EDF). 
If no feasible schedule exists, return "No feasible schedule". 

Show your work in <think> </think> tags. 
And return the final answer of job_id in <answer> </answer> tags, for example <answer> [0, 1, 2] </answer>.
Assistant: Let me solve this step by step.
<think>"""
    
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prompt = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.<|im_end|>\n
<|im_start|>user\nUsing the given jobs {jobs_description} (each as (processing_time, deadline)), create a feasible schedule that completes all jobs before their deadlines using Earliest Deadline First (EDF). 
If no feasible schedule exists, return "No feasible schedule". 

Show your work in <think> </think> tags. 
And return the final answer of job_id in <answer> </answer> tags, for example <answer> [0, 1, 2] </answer>.<|im_end|>\n
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""

    return prompt

def gen_scheduling_dataset(num_samples: int, num_jobs: int = 10, seed_value: int = 42):
    """
    生成排程問題資料集
    Args:
        num_samples: 資料筆數
        num_jobs: 每個樣本中作業數量
        seed_value: 隨機種子
    Returns:
        包含每筆資料 (jobs, schedule) 的列表
    """
    random.seed(seed_value)
    samples = []
    for _ in tqdm(range(num_samples)):
        jobs = generate_hard_edf_instance(num_jobs)
        schedule = compute_edf_schedule(jobs)
        samples.append({
            "jobs": jobs,
            "schedule": schedule
        })
    return samples

def make_map_fn(split, template_type='base'):
    def process_fn(example, idx):
        prompt = generate_scheduling_prompt(example['jobs'], template_type)
        solution = {
            "jobs": example['jobs'],
            "schedule": example['schedule']
        }
        data = {
            "data_source": "scheduling",
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "ability": "scheduling",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                "split": split,
                "index": idx,
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/scheduling', help="本地資料儲存目錄")
    parser.add_argument('--hdfs_dir', default=None, help="HDFS 資料儲存目錄")
    parser.add_argument('--num_samples', type=int, default=10000, help="總樣本數")
    parser.add_argument('--num_jobs', type=int, default=10, help="每筆樣本中的作業數量")
    parser.add_argument('--train_size', type=int, default=8000, help="訓練集大小")
    parser.add_argument('--test_size', type=int, default=2000, help="測試集大小") 
    parser.add_argument('--seed_value', type=int, default=42, help="隨機種子")
    parser.add_argument('--template_type', type=str, default='base', choices=['base', 'qwen-instruct'], help="提示詞模板類型")
    
    args = parser.parse_args()
    
    # 確保樣本數足夠分割成訓練集與測試集
    assert args.num_samples >= args.train_size + args.test_size, "num_samples 必須大於等於 train_size + test_size"
    
    # 生成原始資料集樣本
    raw_samples = gen_scheduling_dataset(num_samples=args.num_samples, num_jobs=args.num_jobs, seed_value=args.seed_value)
    raw_dataset = Dataset.from_list(raw_samples)
    
    # 切分訓練集與測試集
    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))
    
    # 轉換為最終格式
    train_dataset = train_dataset.map(function=make_map_fn('train', args.template_type), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', args.template_type), with_indices=True)
    
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
