from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
import torch
import torch.nn as nn
import ast
import torch.optim as optim
import pandas as pd
import copy
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanSquaredError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获得matbench的数据集
mb = MatbenchBenchmark(autoload=False)
# 打开其中的一个任务
task = MatbenchTask("matbench_phonons")
fold_number = 0
train_data_matbench = task.get_train_and_val_data(fold_number)
X, y = train_data_matbench

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


def get_triplets(structures):
    all_tensor_data = []  # 存储所有结构的三元组数据
    for structure in structures:
        tensor_data = []
        num_atoms = len(structure)
        
        # if num_atoms == 1:
        #     lattice = structure.lattice
        #     for _ in range(3):
        #         triplet_data = (0, 0, lattice.a)
        #         tensor_data.append(triplet_data)
        #     all_tensor_data.append(tensor_data)
        #     continue
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

    max_length = max(len(sublist) for sublist in all_tensor_data)

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
X = X.to(device)

class BondExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionRBF, self).__init__()
        self.num_features = num_features
        self.gamma = gamma

    def forward(self, bond_dist: torch.Tensor) -> torch.Tensor:
        # 生成特征中心张量并确保位于与 bond_dist 相同的设备上
        feature_centers = torch.arange(1, self.num_features + 1).float().to(bond_dist.device)

        # 计算每个距离到各个特征中心的欧几里得距离
        distance_to_centers = torch.abs(feature_centers - bond_dist.unsqueeze(-1)) * 0.5
        # 使用高斯径向基函数计算每个距离对应的特征值
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2).squeeze()

        return rbf_values


class MyModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.embedding_dim = 10
        self.embedding = nn.Embedding(118, self.embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=10)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()
        self.train_metrics = MeanSquaredError()
        self.val_metrics = MeanSquaredError()
        self.test_metrics = MeanSquaredError()

    # def forward(self, structures):
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     embedded_data = []
    #     zero_tensor = torch.zeros(10, device=device)
    #     for triplet_data in structures:
    #         embedded_triplet_data = []

    #         for only_data in triplet_data:
    #             embedding_i = self.embedding(only_data[0].to(torch.long)) if torch.any(
    #                 only_data[0] != 0) else zero_tensor
    #             embedding_j = self.embedding(only_data[1].to(torch.long)) if torch.any(
    #                 only_data[1] != 0) else zero_tensor

    #             if torch.all(embedding_i == zero_tensor):
    #                 distance_tensor = zero_tensor
    #             else:
    #                 distance_tensor = self.bond_expansion(only_data[2])

    #             concatenated_embedding = torch.cat((embedding_i, embedding_j, distance_tensor), dim=-1)
    #             embedded_triplet_data.append(concatenated_embedding)

    #         embedded_triplet_data = torch.stack(embedded_triplet_data)
    #         embedded_data.append(embedded_triplet_data)

    #     embedded_data = torch.stack(embedded_data)

    #     x = embedded_data.view(embedded_data.shape[0], -1)
    #     x = self.relu(self.linear1(x))
    #     x = self.relu(self.linear2(x))
    #     x = self.linear3(x)

    #     return x.squeeze(1)
    
    def forward(self, structures):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        embedded_data = []
        zero_tensor = torch.zeros(10, device=device)

        for triplet_data in structures:
            embedded_triplet_data = []  # 用于存储当前三元组的嵌入数据

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

            embedded_triplet_data = torch.stack(embedded_triplet_data)  # 将当前三元组的嵌入数据堆叠起来
            embedded_data.append(embedded_triplet_data)

        embedded_data = torch.stack(embedded_data)
        # print(embedded_data[0][0])

        # print("-"*100)

        # # 使用第一个自注意力层
        # x, _ = self.multihead_attention1(embedded_data, embedded_data, embedded_data)
        
        # # 使用第二个自注意力层
        # x, _ = self.multihead_attention2(x, x, x)

        #  # 计算每个样本的平均值
        # x = embedded_data.mean(dim=1)
        
        x = embedded_data.view(embedded_data.shape[0], -1)
        # print(x[0][:300])
        # 通过隐藏层1和激活函数1
        x = self.relu(self.linear1(x[:,:1350]))

        # 通过隐藏层2和激活函数2
        x = self.relu(self.linear2(x))

        # 输出层
        x = self.linear3(x)

        x = x.squeeze(1)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0007)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.float())
        loss = self.loss(outputs, targets.float())
        self.train_metrics(outputs, targets.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.float())
        val_loss = self.loss(outputs, targets.float())
        self.val_metrics(outputs, targets.float())
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs.float())
        test_loss = self.loss(outputs, targets.float())
        self.test_metrics(outputs, targets.float())
        self.log('test_loss', test_loss)

class LossPrinterCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch}: Train Loss: {pl_module.train_loss}, Val Loss: {pl_module.val_loss}")


# 输入数据的维度为30，输出数据的维度为1
input_size = 30 * 45
num_heads = 2
hidden_size = 64
output_size = 1

# 实例化模型
model = MyModel(input_size, hidden_size, output_size)
# 将模型移动到 GPU
model.to(device)
print(X.shape)
print("Model is training on device:", device)
# 计算训练集、验证集和测试集的样本数量
total_samples = len(X)
train_samples = int(0.6 * total_samples)
val_samples = int(0.2 * total_samples)
# 测试集的样本数
test_samples = total_samples - train_samples - val_samples

# 划分训练集、验证集和测试集
train_data = X[:train_samples]
train_targets = y[:train_samples]

val_data = X[train_samples:train_samples + val_samples]
val_targets = y[train_samples:train_samples + val_samples]

test_data = X[train_samples + val_samples:]
test_targets = y[train_samples + val_samples:]

train_targets = torch.tensor(train_targets.values, device=device)
val_targets = torch.tensor(val_targets.values, device=device)
test_targets = torch.tensor(test_targets.values, device=device)

# 将目标张量的数据类型转换为torch.float32
train_targets = train_targets.float()
val_targets = val_targets.float()
test_targets = test_targets.float()

# 实例化训练集、验证集和测试集
train_set = torch.utils.data.TensorDataset(train_data, train_targets)
val_set = torch.utils.data.TensorDataset(val_data, val_targets)
test_set = torch.utils.data.TensorDataset(test_data, test_targets)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# 定义 Early Stopping 回调
early_stop_callback = EarlyStopping(monitor='val_loss', patience=20)

# 实例化 Trainer
trainer = pl.Trainer(
    accelerator="gpu", 
    devices = [0],
    max_epochs=1000,
    callbacks=[early_stop_callback, LossPrinterCallback()] if early_stop_callback else [LossPrinterCallback()],
    log_every_n_steps=1,  # 每个 step 记录一次指标
    logger=True,  # 启用默认的 TensorBoardLogger
)


# 训练模型
trainer.fit(model, train_loader, val_loader)

# 测试模型
trainer.test(dataloaders=test_loader)