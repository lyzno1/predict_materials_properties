from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.distributed as dist
from TripletStats import TripletStats
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
from rich.console import Console
from rich.text import Text
from rich.table import Table
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import subprocess
import copy
from collections import defaultdict
import os
import argparse

# from torch_geometric.nn import SchNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        feature_centers = torch.arange(1, self.num_features + 1).float().to(bond_dist.device)

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
  
class MultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MultiLayerGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList([ExponentialGatingGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            for i, gru_cell in enumerate(self.gru_cells):
                h_t[i] = gru_cell(x[:, t, :] if i == 0 else h_t[i-1], h_t[i])
        
        return h_t[-1].unsqueeze(1)
      
class GRUTransformer(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, n_heads, n_layers, activation,
                 ffn_hidden, output_dim, dropout, embedding_dim, num_features, gamma, num_gru_layers):
        super(GRUTransformer, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=num_features, gamma=gamma)
        self.multi_layer_gru = MultiLayerGRU(input_dim, gru_hidden_dim, num_gru_layers)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=gru_hidden_dim, nhead=n_heads, activation=activation,
                                                            dim_feedforward=ffn_hidden, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, output_dim),
        )

    def forward(self, x):
        atom1 = self.embedding(x[:, :, 0].to(torch.long))
        atom2 = self.embedding(x[:, :, 1].to(torch.long))
        bond = self.bond_expansion(x[:, :, 2].float())

        x = torch.cat([atom1, atom2, bond], dim=-1)
        
        gru_out = self.multi_layer_gru(x)
        transformer_out = self.transformer(gru_out)
        transformer_out = self.dropout(transformer_out)
        output = self.fc(transformer_out[:, -1, :])
        return output

class Predictor(pl.LightningModule):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.save_hyperparameters(hparams)
        activation_function = getattr(F, hparams.activation)
        self.model = GRUTransformer(input_dim=hparams.embedding_dim * 2 + hparams.num_features, 
                                    gru_hidden_dim=hparams.hidden_dim, 
                                    ffn_hidden= hparams.ffn_hidden,
                                    n_heads=hparams.n_heads, 
                                    n_layers=hparams.n_layers,
                                    activation=activation_function,
                                    num_gru_layers=hparams.gru_layers,
                                    dropout=hparams.dropout,
                                    embedding_dim=hparams.embedding_dim, 
                                    num_features=hparams.num_features, 
                                    gamma=hparams.gamma, 
                                    output_dim=hparams.output_dim)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target'].float()
        pred = self.model(x)
        loss = F.l1_loss(pred.squeeze(), y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target'].float()
        pred = self.model(x)
        val_loss = F.l1_loss(pred.squeeze(), y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target'].float()
        pred = self.model(x)
        test_loss = F.l1_loss(pred.squeeze(), y)
        self.log('test_mae', test_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
        
class DistributedEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # 每个设备计算其监控指标
        monitor_val = trainer.callback_metrics.get(self.monitor)
        
        # 所有设备同步监控指标，选择最优值
        best_monitor_val = torch.tensor(monitor_val).to(pl_module.device)
        dist.all_reduce(best_monitor_val, op=dist.reduce_op.MIN)
        
        # 检查是否达到early stopping条件
        if monitor_val == best_monitor_val.item():
            super().on_validation_end(trainer, pl_module)

class DistributedModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        # 每个设备计算其监控指标
        monitor_val = trainer.callback_metrics.get(self.monitor)
        
        # 所有设备同步监控指标，选择最优值
        best_monitor_val = torch.tensor(monitor_val).to(pl_module.device)
        dist.all_reduce(best_monitor_val, op=dist.reduce_op.MIN)
        
        # 如果当前设备的监控指标是最优的，则保存模型
        if monitor_val == best_monitor_val.item():
            super().on_validation_end(trainer, pl_module)

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
    with open('./mb_results/xgtpl.txt', 'a') as f:
        if f.tell() != 0:
            f.write('\n' + '*' * 100 + '\n')
        f.write(f"{dataset_name[9:]}, Average MAE: {average_mae}\n")
        # for fold_num, mae in enumerate(results_list):
        #     f.write(f"Fold {fold_num}, MAE:{mae}\n")

    results_list.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xgt')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--n_heads', type=int, default=2, help='number of heads in multi-head attention')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers in transformer')
    parser.add_argument('--activation', type=str, default='silu', help='activation function of Transformer ffn')
    parser.add_argument('--ffn_hidden', type=int, default=1024, help='hidden dimension in feed-forward network')
    parser.add_argument('--gru_layers', type=int, default=4, help='number of layers in GRU')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimension')
    parser.add_argument('--num_features', type=int, default=10, help='number of features')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value for RBF')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--cutoff', type=int, default=1, help='Cutoff length for triplets')
    parser.add_argument('--init_seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

    hparams = parser.parse_args()

    pl.seed_everything(hparams.init_seed)

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            "matbench_jdft2d",  # 636
            "matbench_phonons",  # 1,265
            "matbench_dielectric",  # 4,764
            "matbench_log_gvrh",  # 10,987
            "matbench_log_kvrh",  # 10,987
            # "matbench_perovskites",  # 1w8
            # "matbench_mp_gap",   # 回归 10.6w
            # "matbench_mp_e_form",  # 回归 13w
        ]
    )

    mae_list = []
    for task in mb.tasks:
        task.load()
        dataset_name = task.dataset_name
        for fold in task.folds:

            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            max_length = TripletStats(train_inputs).get_average() * hparams.cutoff # 用于截断/补齐

            x_input = torch.tensor(get_triplets(train_inputs, max_length))  # 处理输入

            dataset = StructureDataset(x_input, train_outputs)
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=hparams.init_seed)

            train_loader = DataLoader(train_data, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                                      persistent_workers=True)
            val_loader = DataLoader(val_data, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                                    persistent_workers=True)

            model = Predictor(hparams)

            early_stop_callback = DistributedEarlyStopping(monitor="val_loss", min_delta=0.00, patience=300, verbose=True, mode="min")
            checkpoint_callback = DistributedModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
            progress_bar = RichProgressBar()
            callbacks=[checkpoint_callback, progress_bar]

            trainer = Trainer(max_epochs=1000,
                                 callbacks=callbacks,
                                 log_every_n_steps=50,
                                 accelerator='gpu',
                                 strategy=DDPStrategy(find_unused_parameters=True)
                                 )
            trainer.fit(model, train_loader, val_loader)

            # 加载验证损失最小的模型权重
            best_model_path = checkpoint_callback.best_model_path
            best_model = Predictor.load_from_checkpoint(best_model_path, hparams=hparams)

            # 测试
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs = torch.tensor(get_triplets(test_inputs, max_length))
            test_dataset = StructureDataset(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers,
                                     persistent_workers=True)

            predict = trainer.test(best_model, test_loader)

            mae = predict[0]['test_mae']
            mae_list.append(mae)

        visualize_results(mae_list, dataset_name)
