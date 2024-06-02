import matplotlib.pyplot as plt
import numpy as np

# 距离的变化范围
distances = np.linspace(0, 5, 500)

# 不同的 gamma 值
gamma_values = [0.1, 1, 2, 10]

plt.figure(figsize=(10, 6))

for gamma in gamma_values:
    rbf_values = np.exp(-gamma * distances ** 2)
    plt.plot(distances, rbf_values, label=f'gamma = {gamma}')

plt.xlabel('Distance')
plt.ylabel('RBF Value')
plt.title('RBF Function with Different Gamma Values')
plt.legend()
plt.grid(True)
plt.show()
