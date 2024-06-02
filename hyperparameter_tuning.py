import subprocess
import os

# 定义超参数组合
batch_size_list = [32, 64]
gamma_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

for batch_size in batch_size_list:
    for gamma in gamma_list:
        command = [
            "python", "main.py",
            "--batch_size", str(batch_size),
            "--gamma", str(gamma),
        ]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, text=True)
