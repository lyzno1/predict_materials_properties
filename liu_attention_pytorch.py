from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import subprocess
import copy
from collections import defaultdict
import os
import argparse
import uuid
from rich.console import Console
from rich.text import Text
from rich.table import Table

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

class TripletStats:
    def __init__(self, structures):
        self.structures = structures
        self.triplet_counts = self.calculate_triplet_counts()
        self.average = self.calculate_average()
        self.max_value = max(self.triplet_counts)
        self.min_value = min(self.triplet_counts)
        self.median = self.calculate_median()
        self.most_common = self.calculate_most_common()
        self.least_common = self.calculate_least_common()
        self.new_max, self.new_min = self.calculate_trimmed_extremes()

    def calculate_triplet_counts(self):
        triplet_counts = []
        for structure in self.structures:
            len_triplet = (len(structure) * (len(structure) - 1)) // 2
            triplet_counts.append(len_triplet)
        return triplet_counts

    def calculate_average(self):
        return sum(self.triplet_counts) / len(self.triplet_counts)

    def calculate_median(self):
        sorted_counts = sorted(self.triplet_counts)
        n = len(sorted_counts)
        if n % 2 == 1:
            return sorted_counts[n // 2]
        else:
            return (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2

    def calculate_most_common(self):
        from collections import Counter
        count = Counter(self.triplet_counts)
        return count.most_common(1)[0][0]

    def calculate_least_common(self):
        from collections import Counter
        count = Counter(self.triplet_counts)
        return count.most_common()[-1][0]

    def calculate_trimmed_extremes(self):
        trimmed_counts = [x for x in self.triplet_counts if x != self.max_value and x != self.min_value]
        if trimmed_counts:
            new_max = max(trimmed_counts)
            new_min = min(trimmed_counts)
            return new_max, new_min
        else:
            return None, None  # 当所有值都相同时，去除后为空列表

    def get_max_value(self):
        print("最大值:", self.max_value)
        return int(self.max_value)

    def get_min_value(self):
        print("最小值:", self.min_value)
        return int(self.min_value)

    def get_median(self):
        print("中位数:", self.median)
        return int(self.median)

    def get_average(self):
        print("平均数:", self.average)
        return int(self.average)

    def get_most_common(self):
        print("出现最多的数:", self.most_common)
        return int(self.most_common)

    def get_least_common(self):
        print("出现最少的数:", self.least_common)
        return int(self.least_common)

    def get_new_max(self):
        print("去除最大最小值之后的最大值:", self.new_max)
        return int(self.new_max)

    def get_new_min(self):
        print("去除最大最小值之后的最小值:", self.new_min)
        return int(self.new_min)

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

class ExponentialGatingGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExponentialGatingGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_dim, 3 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim)
    
    def forward(self, x, hidden):
        preact = self.i2h(x) + self.h2h(hidden)
        
        gates = preact[:, :2 * self.hidden_dim]
        gates = torch.sigmoid(gates)
        r_gate, z_gate = gates.chunk(2, 1)
        
        # Exponential gating
        r_gate = torch.exp(-torch.exp(-r_gate))
        z_gate = torch.exp(-torch.exp(-z_gate))
        
        h2h_output = self.h2h(hidden)
        n_t_preact = preact[:, 2 * self.hidden_dim:] + r_gate * h2h_output[:, 2 * self.hidden_dim:]
        
        n_t = torch.tanh(n_t_preact)
        h_t = (1 - z_gate) * n_t + z_gate * hidden
        
        return h_t
    
class GRUTransformer(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, transformer_hidden_dim, n_heads, n_layers, output_dim, embedding_dim, dropout):
        super(GRUTransformer, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=16, gamma=1.0)
        self.gru_cell = ExponentialGatingGRUCell(input_dim, gru_hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=gru_hidden_dim, nhead=n_heads, dim_feedforward=transformer_hidden_dim, 
                                                            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=n_layers)
        self.fc = nn.Linear(gru_hidden_dim, output_dim)

    def forward(self, x):
        atom1 = self.embedding(x[:, :, 0].to(torch.long))
        atom2 = self.embedding(x[:, :, 1].to(torch.long))
        bond = self.bond_expansion(x[:, :, 2].float())
        x = torch.cat([atom1, atom2, bond], dim=-1)

        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.gru_cell.hidden_dim).to(x.device)
        
        gru_out = []
        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)
            gru_out.append(h_t.unsqueeze(1))
        
        gru_out = torch.cat(gru_out, dim=1)
        transformer_out = self.transformer(gru_out)
        output = self.fc(transformer_out[:, -1, :])
        return output

class Attention_structure_model(nn.Module):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0.2):
        super(Attention_structure_model, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=10, gamma=1.0)

        self.gru = nn.GRU(input_size=embedding_dim * 3, hidden_size=hidden_size, batch_first=True, num_layers=2,
                          dropout=dropout)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size)
        )

        self.self_attention1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True, dropout=0.2)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

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

        # 调用 self-attention
        attention_output, _ = self.self_attention(gru_output, gru_output, gru_output)
        attention_output = self.layer_norm1(gru_output + attention_output)

        feed_forward_output = self.ffn(attention_output)
        feed_forward_output = self.layer_norm2(attention_output + feed_forward_output)

        attention_output, _ = self.self_attention(feed_forward_output, feed_forward_output, feed_forward_output)
        attention_output = self.layer_norm3(attention_output + attention_output)

        feed_forward_output = self.ffn(attention_output)
        attention_output = self.layer_norm4(attention_output + feed_forward_output)

        # 调用 self-attention1
        attention_output, _ = self.self_attention1(attention_output, attention_output, attention_output)

        output = attention_output[:, -1, :]  # 选择最后一个时间步的隐藏状态

        output = self.out_layer(output)

        return output.squeeze(1)

class liu_attention(nn.Module):
    def __init__(self, embedding_dim=16, hidden_size=64, output_size=1, n_heads=2, n_layers=2, ffn_hidden=1024, dropout=0.2):
        super(liu_attention, self).__init__()
        # self.model = Attention_structure_model(embedding_dim=embedding_dim, hidden_size=hidden_size,
        #                                       output_size=output_size, dropout=dropout)

        self.model = GRUTransformer(input_dim=embedding_dim * 3, gru_hidden_dim=hidden_size, transformer_hidden_dim=ffn_hidden, 
                                    output_dim=output_size, n_heads=n_heads, n_layers=n_layers, 
                                    dropout=dropout, embedding_dim=embedding_dim)
    def forward(self, x):
        return self.model(x)


class liu_attention_trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
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
    def __init__(self, 
                 patience: int = 100,
                 min_delta: float = 0.00, 
                 path: str = None, 
                 monitor: str = 'val_loss', 
                 mode: str = 'min', 
                 save_top_k: int = 1,
                 dataset_name: str = None,
                 save_model: bool = False
                 ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model.
            monitor (str): Quantity to be monitored. Default is 'val_loss'.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing.
            save_top_k (int): Number of best models to save. Default is 1.
            dataset_name (str): Name of the dataset. Default is None.
            save_model (bool): Whether to save the model. Default is False.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metrics = [float('inf')] if mode == 'min' else [float('-inf')]
        self.counter = 0
        self.early_stop = False
        self.dataset_name = dataset_name
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.is_better = None
        self.sym = False
        self.save_model = save_model
        self.run_id = str(uuid.uuid4())  # Unique identifier for the training run

        if path:
            self.path = path
            self.sym = False
        else:
            self.path = None

        if self.mode == 'min':
            self.is_better = lambda a, best: a < best
        elif self.mode == 'max':
            self.is_better = lambda a, best: a > best
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def __call__(self, val_metric, model):
        if self.is_better(val_metric, self.best_metrics[-1] - self.min_delta):
            self.best_metrics.append(val_metric)
            if self.save_model:
                self.save_checkpoint(model)
            print(f"{self.monitor} improved to {val_metric:.4f}")
            self._cleanup_checkpoints()
            self.counter = 0 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        best_metric = min(self.best_metrics) if self.mode == 'min' else max(self.best_metrics)
        print(f"Current best {self.monitor}: {best_metric:.4f}")

    def add_path(self):
        if self.path is None:
            self.path = f'./pytorch_checkpoints/{self.dataset_name}/{self.monitor}-{self.best_metrics[-1]:.4f}-{self.run_id}.pth'
            self.sym = True
        if self.sym:
            self.path = f'./pytorch_checkpoints/{self.dataset_name}/{self.monitor}-{self.best_metrics[-1]:.4f}-{self.run_id}.pth'
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save_checkpoint(self, model):
        """Saves model when monitored metric improves."""
        self.add_path()
        torch.save(model.state_dict(), self.path)

    def _cleanup_checkpoints(self):
        """Deletes all but the top k models for the current run instance."""
        checkpoints = glob.glob(f'./pytorch_checkpoints/{self.dataset_name}/{self.monitor}-*-{self.run_id}.pth')
        checkpoints.sort(key=os.path.getmtime)
        if len(checkpoints) > self.save_top_k:
            for ckpt in checkpoints[:-self.save_top_k]:
                os.remove(ckpt)

    def load_checkpoint(self, model):
        """Loads the saved model."""
        model.load_state_dict(torch.load(self.path))




def visualize_results(results_list, dataset_name):
    print('-' * 100)
    console = Console()
    table = Table(title=f"{dataset_name[9:]} Results")
    table.add_column("Fold", style="cyan")
    table.add_column("MAE", style="magenta")

    for i, mae in enumerate(results_list):
        table.add_row('fold'+str(i), f"{mae:.4f}")

    average_mae = sum(results_list) / len(results_list)
    table.add_row("[bold]Average[/]", f"[bold magenta]{average_mae:.4f}[/]")

    console.print(table)

    # 写入结果到文件
    with open('./mb_results/xgt.txt', 'a') as f:
        if f.tell() != 0:
            f.write('\n')
        f.write(f"{dataset_name[9:]}, Average MAE: {average_mae}\n")
        for fold_num, mae in enumerate(results_list):
            f.write(f"Fold {fold_num}, MAE:{mae}\n")

    results_list.clear()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp.set_start_method('spawn')
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
            "matbench_jdft2d",  # 636
            # "matbench_phonons",  # 1,265
            # "matbench_dielectric",  # 4,764
            "matbench_log_gvrh",  # 10,987
            "matbench_log_kvrh",  # 10,987
            "matbench_perovskites",  # 1w8
            # "matbench_mp_gap",   # 回归 10.6w
            # "matbench_mp_e_form",  # 回归 13w
        ]
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(2)
    # 保存每个fold的MAE
    mae_list = []

    for task in mb.tasks:
        task.load()
        dataset_name = task.dataset_name
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            max_length = TripletStats(train_inputs).get_average()  # 用于截断/补齐
            x_input = torch.tensor(get_triplets(train_inputs, max_length)).to(device)  # 处理输入

            dataset = StructureDataset(x_input, train_outputs)
            train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
            train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
            batch_size = args.batch_size

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x, num_workers=4)


            model = liu_attention().to(device)

            # 初始化训练器
            trainer = liu_attention_trainer(model)
            early_stopping = EarlyStopping(patience=300, min_delta=0.000, save_model=True, dataset_name=dataset_name)
            num_epochs = 1000

            # 训练循环
            for epoch in range(num_epochs):
                early_stopping.epoch = epoch
                train_losses = []
                model.train()
                for batch in train_loader:
                    inputs = torch.stack([torch.tensor(item['input']).clone().detach().requires_grad_(True) for item in batch]).to(device).squeeze()
                    targets = torch.stack([torch.tensor(item['target']).clone().detach() for item in batch]).to(device)
                    if targets.dim() == 1:
                        targets = targets.view(-1, 1)
                    loss = trainer.train_step(inputs, targets)
                    train_losses.append(loss)

                # 计算平均训练损失
                avg_train_loss = sum(train_losses) / len(train_losses)

                # 验证循环
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = torch.stack([torch.tensor(item['input']).clone().detach().requires_grad_(True) for item in batch]).to(device).squeeze()
                        targets = torch.stack([torch.tensor(item['target']).clone().detach() for item in batch]).to(device)
                        if targets.dim() == 1:
                            targets = targets.view(-1, 1)
                        val_loss = trainer.val_step(inputs, targets)
                        val_losses.append(val_loss)

                # 计算平均验证损失
                avg_val_loss = sum(val_losses) / len(val_losses)

                # 更新 EarlyStopping
                early_stopping(avg_val_loss, model)
                print('Epoch: [{epoch}][{num_epochs}]\t'
                    'Loss {avg_train_loss:.4f}\t'
                    'val_Loss {avg_val_loss:.4f}\n'.format(
                    epoch=epoch + 1, num_epochs=num_epochs,
                    avg_train_loss=avg_train_loss,
                    avg_val_loss=avg_val_loss)
                )

                if early_stopping.early_stop:
                    if epoch == num_epochs:
                        print("max epoch reached")
                    else:
                        print("Early stopping")
                    break

            # 在训练结束后，加载最佳模型参数
            if early_stopping.save_model:
                early_stopping.load_checkpoint(model)

            # 测试循环
            test_losses = []
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs = torch.tensor(get_triplets(test_inputs, max_length)).to(device)
            test_dataset = StructureDataset(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=lambda x: x, num_workers=4)
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    inputs = torch.stack([torch.tensor(item['input']).clone().detach().requires_grad_(True) for item in batch]).to(device).squeeze()
                    targets = torch.stack([torch.tensor(item['target']).clone().detach() for item in batch]).to(device)
                    if targets.dim() == 1:
                        targets = targets.view(-1, 1)
                    test_loss = trainer.test_step(inputs, targets)
                    test_losses.append(test_loss)

            # 计算平均测试损失
            avg_test_loss = sum(test_losses) / len(test_losses)
            mae_list.append(avg_test_loss)
            console = Console()
            text = Text(f'Test MAE: {avg_test_loss:.4f}', style="bold red")
            console.print(text)

        visualize_results(mae_list, dataset_name)




