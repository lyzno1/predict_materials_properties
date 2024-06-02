from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import subprocess
import copy
from collections import defaultdict
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='liu_attention')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
# parser.add_argument('--e', type=int, default=0, help='number of node embedding dim')

args = parser.parse_args()


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
def get_atomic_number(symbol): #元素符号转化为原子序数
    return Element(symbol).number


def tripletshu(structures):  # 获得数据集中平均长度
    sum = 0
    for structure in structures:
        len_triplet = (len(structure) * (len(structure) - 1))/2
        sum += len_triplet
    # max_length = max(len(structure) for structure in structures)  # 获取所有结构中的最大长度
    # max_tripletshu = ((max_length - 1) * max_length) / 2
    # return int(max_tripletshu)
    average = sum/len(structures)
    return int(average)


def get_triplets(structures, max_length):  # 处理成三元组
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

                # 将原子转换为对应的原子序数
                atomic_number_i = get_atomic_number(element_i)
                atomic_number_j = get_atomic_number(element_j)
                # 存储原始的三元组数据
                triplet_data = (atomic_number_i, atomic_number_j, distance)
                tensor_data.append(triplet_data)

        # 对三元组列表按照最后一个元素（距离信息）进行升序排序
        tensor_data.sort(key=lambda x: x[2], reverse=False)

        # 截断数据到max_length长度
        if len(tensor_data) > max_length:
            tensor_data = tensor_data[:max_length]
        # 将当前结构的三元组数据添加到总列表中
        all_tensor_data.append(tensor_data)

    # 对不足最大长度的子列表进行补充
    for sublist in all_tensor_data:
        while len(sublist) < max_length:
            sublist.append((0, 0, 0.0))

    return all_tensor_data




class StructureDataset(Dataset):
    def __init__(self, structure, target):
        self.input = structure
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target.iloc[idx]

        # 返回合金描述和目标值
        return {'input': input, 'target': target}

# 距离bond expansion处理
class BondExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionRBF, self).__init__()
        self.num_features = num_features
        self.gamma = gamma


    def forward(self, bond_dist: torch.Tensor) -> torch.Tensor:
        # 生成特征中心张量
        feature_centers = torch.arange(1, self.num_features + 1).float().to(device)

        # 计算每个距离到各个特征中心的欧几里得距离
        distance_to_centers = torch.abs(feature_centers - bond_dist.unsqueeze(-1))

        # 使用高斯径向基函数计算每个距离对应的特征值
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2).squeeze()

        return rbf_values

class Attention_structer_model(nn.Module):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0.2):
        super(Attention_structer_model, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=10, gamma=1.0)

        self.gru = nn.GRU(input_size=embedding_dim * 3, hidden_size=hidden_size, batch_first=True, num_layers=2,
                          dropout=dropout)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)

        self.linear_feed = nn.Linear(hidden_size, 1024)
        self.relu1 = nn.ReLU()
        self.linear_forward = nn.Linear(1024, hidden_size)

        self.self_attention1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 32)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32, output_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.layer_norm4 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # 嵌入元素和键扩展
        atom1 = self.embedding(x[:, :, 0].to(torch.long))
        atom2 = self.embedding(x[:, :, 1].to(torch.long))
        bond = self.bond_expansion(x[:, :, 2].float())
        embedded_data = torch.cat([atom1, atom2, bond], dim=-1)

        # GRU 输入数据格式应为 (batch_size, seq_len, input_size)
        gru_output, _ = self.gru(embedded_data)

        # # 调用 self-attention
        # attention_output, _ = self.self_attention(gru_output, gru_output, gru_output)
        # attention_output = self.linear_feed(attention_output)
        # attention_output = self.relu1(attention_output)
        # attention_output = self.linear_forward(attention_output)
        # attention_output, _ = self.self_attention(attention_output, attention_output, attention_output)
        # attention_output = self.linear_feed(attention_output)
        # attention_output = self.relu1(attention_output)
        # attention_output = self.linear_forward(attention_output)
        # attention_output, _ = self.self_attention1(attention_output, attention_output, attention_output)


        # 调用 self-attention
        attention_output, _ = self.self_attention(gru_output, gru_output, gru_output)
        attention_output = self.layer_norm1(gru_output + attention_output)

        feed_forward_output = self.linear_feed(attention_output)
        feed_forward_output = self.relu1(feed_forward_output)
        feed_forward_output = self.linear_forward(feed_forward_output)
        attention_output = self.layer_norm2(attention_output + feed_forward_output)

        attention_output, _ = self.self_attention(attention_output, attention_output, attention_output)
        attention_output = self.layer_norm3(attention_output + attention_output)

        feed_forward_output = self.linear_feed(attention_output)
        feed_forward_output = self.relu1(feed_forward_output)
        feed_forward_output = self.linear_forward(feed_forward_output)
        attention_output = self.layer_norm4(attention_output + feed_forward_output)

        # 调用 self-attention1
        attention_output, _ = self.self_attention1(attention_output, attention_output, attention_output)

        output = attention_output[:, -1, :]  # 选择最后一个时间步的隐藏状态

        # 线性层
        output = self.linear1(self.relu2(self.linear(output)))

        output = output.squeeze(1)

        return output

class liu_attention(nn.Module):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0):
        super(liu_attention, self).__init__()
        self.model = Attention_structer_model(embedding_dim=embedding_dim, hidden_size=hidden_size,
                                              output_size=output_size, dropout=dropout)

    def forward(self, x):
        return self.model(x)


class liu_attention_trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.L1Loss()

    def train_step(self, x, label):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(x)
        loss = self.loss_fn(predict, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, x, label):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(x)
            val_loss = self.loss_fn(predict, label)
        return val_loss.item()

    def test_step(self, x, label):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(x)
            test_loss = self.loss_fn(predict, label)
        return test_loss.item()


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            print(f"Initial best validation loss set to {val_loss:.4f}")
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        print(f"Current best validation loss: {self.best_loss:.4f}")

    def save_checkpoint(self, model):
        """Saves model when validation loss improves."""
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        """Loads the saved model."""
        model.load_state_dict(torch.load(self.path))

def visualize_results(results_list): # 可视化结果并保存到文件中
    for i, mae in enumerate(results_list):
        print(f"Fold {i} MAE: {mae}")
    average_mae = sum(mae_list) / len(mae_list)
    print(f"Average MAE across all folds: {average_mae}")

    # 写入结果到文件
    # with open('results.txt', 'a') as f:
    #     if f.tell() != 0:
    #         f.write('\n')
    #     for fold_num, mae in enumerate(results_list):
    #         f.write(f"batch_size:{batch_size}, Fold {fold_num}, MAE:{mae}\n")
    #     f.write(f"batch_size:{batch_size}, Average MAE: {average_mae}\n")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_seed = 42
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 获得matbench的数据集
    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            # "matbench_jdft2d",  # 636
            "matbench_phonons",  # 1,265
            # "matbench_dielectric",  # 4,764
            # "matbench_log_gvrh",  # 10,987
            # "matbench_log_kvrh",  # 10,987
            # "matbench_perovskites",  # 1w8
            # "matbench_mp_gap",   # 回归 10.6w
            # "matbench_mp_e_form",  # 回归 13w
        ]
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    # 保存每个fold的MAE
    mae_list = []

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            max_length = tripletshu(train_inputs) * 2  # 用于截断/补齐
            x_input = torch.tensor(get_triplets(train_inputs, max_length)).to(device)  # 处理输入

            dataset = StructureDataset(x_input, train_outputs)
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
            batch_size = args.batch_size
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

            model = liu_attention(embedding_dim=10, hidden_size=64, output_size=1, dropout=0).to(device)

            # 初始化训练器
            trainer = liu_attention_trainer(model)
            early_stopping = EarlyStopping(patience=300, min_delta=0.001, path=f'best_model_fold_{fold}.pth')
            num_epochs = 1500

            # 训练循环
            for epoch in range(num_epochs):
                train_losses = []
                model.train()
                for batch in train_loader:
                    x, label = batch['input'].to(device), batch['target'].float().to(device)
                    train_loss = trainer.train_step(x, label)
                    train_losses.append(train_loss)

                # 计算平均训练损失
                avg_train_loss = sum(train_losses) / len(train_losses)

                # 验证循环
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x, label = batch['input'].to(device), batch['target'].float().to(device)
                        val_loss = trainer.val_step(x, label)
                        val_losses.append(val_loss)

                # 计算平均验证损失
                avg_val_loss = sum(val_losses) / len(val_losses)

                # 更新 EarlyStopping
                early_stopping(avg_val_loss, model)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # 在训练结束后，加载最佳模型参数
            early_stopping.load_checkpoint(model)

            # 测试循环
            test_losses = []
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs = torch.tensor(get_triplets(test_inputs, max_length))
            test_dataset = StructureDataset(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    x, label = batch['input'].to(device), batch['target'].float().to(device)
                    test_loss = trainer.test_step(x, label)
                    test_losses.append(test_loss)

            # 计算平均测试损失
            avg_test_loss = sum(test_losses) / len(test_losses)
            mae_list.append(avg_test_loss)
            print(f'Test Loss: {avg_test_loss:.4f}')
        visualize_results(mae_list)




