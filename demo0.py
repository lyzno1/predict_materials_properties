from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import subprocess
import copy
from collections import defaultdict

if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch
# 获得matbench的数据集
mb = MatbenchBenchmark(autoload=False)
# 打开其中的一个任务
task = MatbenchTask("matbench_phonons")
fold_number = 0
train_data_matbench = task.get_train_and_val_data(fold_number)
test_data_matbench = task.get_test_data(fold_number, include_target=True)
X, y = train_data_matbench
X1, y1 = test_data_matbench


def get_atom_distance(structure, atom_i, atom_j):
    """
        计算两个原子之间的距离
        Args:
            structure (Structure): pymatgen Structure 对象
            atom_i (int): 第一个原子的索引
            atom_j (int): 第二个原子的索引
        Returns:
            distance (float): 两个原子之间的距离
        """
    site_i = structure[atom_i]
    site_j = structure[atom_j]
    distance = site_i.distance(site_j)
    return distance


# 将原子的化学式转化为原子序数
def get_atomic_number(element_symbol):
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
        'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
        'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
        'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
        'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
        'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
        'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
        'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
        'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
        'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
        'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
        'Lv': 116, 'Ts': 117, 'Og': 118
    }
    return atomic_numbers.get(element_symbol, 0)  # 如果找不到元素则返回0


max_distance = float('-inf')
min_distance = float('inf')

def tripletshu(structures):
    max_length = max(len(structure) for structure in structures)  # 获取所有结构中的最大长度
    max_tripletshu = ((max_length-1) * max_length)/2
    return max_tripletshu

print(tripletshu(X),tripletshu(X1))
if tripletshu(X)>tripletshu(X1):
    max_length = tripletshu(X)
else: max_length = tripletshu(X1)
print(max_length)

def get_triplets(structures):
    global max_distance, min_distance
    global max_length
    all_tensor_data = []  # 存储所有结构的三元组数据
    for structure in structures:
        tensor_data = []
        num_atoms = len(structure)

        if num_atoms == 1:
            lattice = structure.lattice
            atom_symbol = structure[0].species_string
            atomic_number = get_atomic_number(atom_symbol)
            triplet_data = (atomic_number, atomic_number, lattice.a)
            tensor_data.append(triplet_data)
            triplet_data = (atomic_number, atomic_number, lattice.b)
            tensor_data.append(triplet_data)
            triplet_data = (atomic_number, atomic_number, lattice.c)
            tensor_data.append(triplet_data)
            all_tensor_data.append(tensor_data)
            continue

        for i in range(num_atoms):
            element_i = structure[i].species_string
            for j in range(i + 1, num_atoms):
                element_j = structure[j].species_string
                distance = get_atom_distance(structure, i, j)
                if distance > max_distance: max_distance = distance
                if distance < min_distance: min_distance = distance

                # 将原子转换为对应的原子序数
                atomic_number_i = get_atomic_number(element_i)
                atomic_number_j = get_atomic_number(element_j)
                # 存储原始的三元组数据
                triplet_data = (atomic_number_i, atomic_number_j, distance)
                tensor_data.append(triplet_data)

        # 对三元组列表按照最后一个元素（距离信息）进行升序排序
        tensor_data.sort(key=lambda x: x[2], reverse=False)

        # 将当前结构的三元组数据添加到总列表中
        all_tensor_data.append(tensor_data)

    # # 寻找第一个元素为零的三元组并返回索引
    # for idx, tensor_data in enumerate(all_tensor_data):
    #     if tensor_data and tensor_data[0][0] == 0:
    #         print(f"Fault in structure at index {idx}, Triplet: {tensor_data}")

    # 对不足最大长度的子列表进行补充
    for sublist in all_tensor_data:
        while len(sublist) < max_length:
            sublist.append((0, 0, 0.0))

    return all_tensor_data


X = get_triplets(X)
X = torch.tensor(X)


X1 = get_triplets(X1)
X1 = torch.tensor(X1)

print("没有全连接层")

# 距离bond expansion处理
class BondExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionRBF, self).__init__()
        self.num_features = num_features
        self.gamma = gamma


    def __call__(self, bond_dist: torch.Tensor) -> torch.Tensor:
        # 生成特征中心张量
        feature_centers = torch.arange(1, self.num_features + 1).float()

        # 计算每个距离到各个特征中心的欧几里得距离
        distance_to_centers = torch.abs(feature_centers - bond_dist.unsqueeze(-1))

        # 使用高斯径向基函数计算每个距离对应的特征值
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2).squeeze()

        return rbf_values

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prop, running_mean, test_running_mean):
        super(MyModel, self).__init__()
        self.embedding_dim = 10
        self.running_mean = running_mean
        self.test_running_mean = test_running_mean
        self.embedding = nn.Embedding(119, self.embedding_dim)
        self.hidden_size = hidden_size
        self.bond_expansion = BondExpansionRBF(num_features=10)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.linear_feed = nn.Linear(hidden_size, 1024)
        self.linear_forward = nn.Linear(1024, hidden_size)
        self.dropout = nn.Dropout(dropout_prop)
        self.bn = nn.BatchNorm1d(running_mean)  # 添加批归一化层
        self.bn1 = nn.BatchNorm1d(test_running_mean)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 32)
        self.linear1 = nn.Linear(32, output_size)
        # self.zyr = nn.Transformer(self.embedding_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
        self.self_attention1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True, dropout=0.2)
    def forward(self, structures):
        embedded_data = []
        zero_tensor = torch.zeros(10)

        for triplet_data in structures:
            embedded_triplet_data = []

            for only_data in triplet_data:
                embedding_i = self.embedding(only_data[0].to(torch.long)) if torch.any(
                    only_data[0] != 0) else zero_tensor
                embedding_j = self.embedding(only_data[1].to(torch.long)) if torch.any(
                    only_data[1] != 0) else zero_tensor

                if torch.all(embedding_i == zero_tensor):
                    distance_tensor = zero_tensor
                else:
                    distance_tensor = self.bond_expansion(only_data[2])

                concatenated_embedding = torch.cat((embedding_i, embedding_j, distance_tensor), dim=-1)
                embedded_triplet_data.append(concatenated_embedding)

            embedded_triplet_data = torch.stack(embedded_triplet_data)
            embedded_data.append(embedded_triplet_data)

        embedded_data = torch.stack(embedded_data)  # shape为(batch_size, seq_length, input_size)
        # embedded_data = embedded_data[:, :100, :]

        # GRU 输入数据格式应为 (batch_size, seq_len, input_size)
        gru_output, _ = self.gru(embedded_data)

        # 调用 self-attention
        attention_output, _ = self.self_attention(gru_output, gru_output, gru_output)
        # attention_output = self.linear_feed(attention_output)
        # attention_output = self.relu(attention_output)
        # attention_output = self.linear_forward(attention_output)
        attention_output, _ = self.self_attention1(attention_output, attention_output, attention_output)
        # if attention_output.size(1) == running_mean:
        #     attention_output = self.bn(attention_output)  # 在自注意力层后应用批归一化
        # elif attention_output.size(1) == test_running_mean:
        #     attention_output = self.bn1(attention_output)
        # else:
        #     attention_output = self.bn2(attention_output)
        # 应用 dropout
        # attention_output = self.dropout(attention_output)
        output = attention_output[:, -1, :]  # 选择最后一个时间步的隐藏状态

        # 线性层
        output = self.linear1(self.relu(self.linear(output)))

        output = output.squeeze(1)


        return output


# 模型参数的初始化
input_size = 30
hidden_size = 64
output_size = 1
dropout_prop = 0
running_mean = X.size(1)
test_running_mean = X1.size(1)

# 实例化模型
model = MyModel(input_size, hidden_size, output_size, dropout_prop, running_mean, test_running_mean)


# 计算训练集、验证集的样本数量
total_samples = len(X)
train_samples = int(0.75 * total_samples)
val_samples = total_samples - train_samples

# 划分训练集、验证集和测试集
train_data = X[:train_samples]
train_targets = y[:train_samples]

val_data = X[train_samples:]
val_targets = y[train_samples:]

test_data = X1
test_targets = y1

train_targets = torch.tensor(train_targets.values)
val_targets = torch.tensor(val_targets.values)
test_targets = torch.tensor(test_targets.values)

# 将目标张量的数据类型转换为torch.float32
train_targets = train_targets.float()
val_targets = val_targets.float()
test_targets = test_targets.float()

# 实例化训练集、验证集和测试集
train_set = torch.utils.data.TensorDataset(train_data, train_targets)
val_set = torch.utils.data.TensorDataset(val_data, val_targets)
test_set = torch.utils.data.TensorDataset(test_data, test_targets)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 获取当前 Commit ID 的函数
def get_current_commit_id():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


# 创建一个空的 DataFrame 用于记录结果
results = []

# 初始化早停参数
early_stopping_patience = 100
current_patience = 0
best_val_loss = float('inf')
best_model_state_dict = None
num_of_iterations = 1000
# 训练循环
for epoch in range(num_of_iterations):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.to()
        targets = targets.float()

        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # 获取当前 Commit ID
    commit_id = get_current_commit_id()

    # 在验证集上计算损失
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to()
            targets = targets.float()
            outputs = model(inputs.float())
            val_loss = criterion(outputs, targets)
            total_val_loss += val_loss.item()

    val_loss = total_val_loss / len(val_loader)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state_dict = copy.deepcopy(model.state_dict())
        torch.save(model, 'best_model fold0.pth')

        current_patience = 0
        # 更新结果列表中对应的最佳模型损失值
        best_model_result_index = None
        for i, result in enumerate(results):
            if result['Commit ID'] == commit_id:
                result['Validation Loss'] = val_loss
                best_model_result_index = i
                break
    else:
        current_patience += 1

    # 打印训练、验证损失
    print(f'Epoch {epoch + 1}/{num_of_iterations}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 检查是否进行早停
    if current_patience >= early_stopping_patience:
        print("Early stopping triggered.")
        break

# 在测试集上评估最佳模型
model.load_state_dict(best_model_state_dict)
model.eval()
total_test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to()
        targets = targets.float()
        outputs = model(inputs.float())
        test_loss = criterion(outputs, targets)
        total_test_loss += test_loss.item()

test_loss = total_test_loss / len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# 将结果列表转换为 DataFrame
results_df = pd.DataFrame.from_records(results)

# 将 DataFrame 写入到 Excel 文件中
results_df.to_excel('model_results.xlsx', index=False)