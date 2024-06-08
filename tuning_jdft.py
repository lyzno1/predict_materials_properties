import subprocess
import os
import random

# 设置随机种子以确保可重复性
# random.seed(42)

# 定义超参数组合
batch_size_list = [32]
gamma_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
sigma_list = [1.0]
# cutoff_range = (1, 30)  # cutoff的范围

# 随机查找
for batch_size in batch_size_list:
    for gamma in gamma_list:
        command = [
            "python", "two_tuple.py",
            "--batch_size", str(batch_size),
            "--gamma", str(gamma),
        ]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, text=True)
