from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import torch.nn.functional as F
from TripletStats import TripletStats
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning import Trainer
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
from scipy.spatial import Voronoi
import scipy

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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

def get_atomic_number(symbol): #元素符号转化为原子序数
    return Element(symbol).number

from scipy.spatial import Voronoi
import numpy as np

def get_triplets_with_angles_as_features(structures, max_length):
    all_tensor_data = []
    for structure in structures:
        tensor_data = []
        num_atoms = len(structure)

        if num_atoms < 5:
            print(f"Structure with {num_atoms} atoms skipped (not enough points for Voronoi diagram).")
            tensor_data.append((0, 0, 0.0, 0.0))
            all_tensor_data.append(tensor_data)
            continue

        positions = np.array([atom.coords for atom in structure])

        try:
            vor = Voronoi(positions)
        except scipy.spatial.qhull.QhullError as e:
            print(f"Voronoi diagram error: {e}")
            tensor_data.append((0, 0, 0.0, 0.0))
            all_tensor_data.append(tensor_data)
            continue

        for i in range(num_atoms):
            element_i = structure[i].species_string
            atomic_number_i = get_atomic_number(element_i)
            neighbors = vor.regions[vor.point_region[i]]

            if -1 in neighbors:
                neighbors.remove(-1)

            for j in neighbors:
                if j < num_atoms and j > i:
                    element_j = structure[j].species_string
                    atomic_number_j = get_atomic_number(element_j)
                    distance_ij = get_atom_distance(structure, i, j)

                    for k in neighbors:
                        if k < num_atoms and k != i:
                            angle_ijk = calculate_angle(structure, i, j, k)
                            triplet_data = (atomic_number_i, atomic_number_j, distance_ij, angle_ijk)
                            tensor_data.append(triplet_data)

        if len(tensor_data) > max_length:
            tensor_data = tensor_data[:max_length]

        all_tensor_data.append(tensor_data)

    for sublist in all_tensor_data:
        while len(sublist) < max_length:
            sublist.append((0, 0, 0.0, 0.0))

    return all_tensor_data

def calculate_angle(structure, i, j, k):
    v1 = structure[j].coords - structure[i].coords
    v2 = structure[k].coords - structure[j].coords

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

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
    
class AngleExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, sigma: float = 1.0):
        super(AngleExpansionRBF, self).__init__()
        self.num_features = num_features
        self.sigma = sigma
        self.centers = torch.linspace(0, 1, num_features)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        angle_features = []
        for center in self.centers:
            feature = torch.exp(-0.5 * ((angles - center) / self.sigma) ** 2)
            angle_features.append(feature)
        angle_features = torch.stack(angle_features, dim=-1)
        return angle_features

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
    def __init__(self, input_dim, gru_hidden_dim, transformer_hidden_dim, n_heads, n_layers, 
                 output_dim, embedding_dim, num_features, gamma):
        super(GRUTransformer, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=num_features, gamma=gamma)
        self.angle_expansion = AngleExpansionRBF(num_features=num_features, sigma=1.0)
        self.gru_cell = ExponentialGatingGRUCell(input_dim, gru_hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=gru_hidden_dim, nhead=n_heads, dim_feedforward=transformer_hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=n_layers)
        self.fc = nn.Linear(gru_hidden_dim, output_dim)

    def forward(self, x):
        atom1 = self.embedding(x[:, :, 0].to(torch.long))
        atom2 = self.embedding(x[:, :, 1].to(torch.long))
        bond = self.bond_expansion(x[:, :, 2].float())
        angle = self.angle_expansion(x[:, :, 3].float())

        x = torch.cat([atom1, atom2, bond, angle], dim=-1)

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
    
class Predictor(pl.LightningModule):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = GRUTransformer(input_dim=hparams.embedding_dim * 2 + hparams.num_features * 2, 
                                    gru_hidden_dim=hparams.hidden_dim, 
                                    transformer_hidden_dim=hparams.hidden_dim, 
                                    n_heads=hparams.n_heads, 
                                    n_layers=hparams.n_layers,
                                    embedding_dim=hparams.embedding_dim, 
                                    num_features=hparams.num_features, 
                                    gamma=hparams.gamma, 
                                    output_dim=hparams.output_dim)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, label = batch['input'], batch['target'].float()
        predict = self.model(x)
        loss = F.l1_loss(predict, label)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch['input'], batch['target'].float()
        predict = self.model(x)
        val_loss = F.l1_loss(predict, label)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, label = batch['input'], batch['target'].float()
        predict = self.model(x)
        test_loss = F.l1_loss(predict, label)
        self.log('test_mae', test_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
        
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
            f.write('\n')
        f.write(f"{dataset_name[9:]}, Average MAE: {average_mae}\n")
        for fold_num, mae in enumerate(results_list):
            f.write(f"Fold {fold_num}, MAE:{mae}\n")

    results_list.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xgt')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--n_heads', type=int, default=2, help='number of heads in multi-head attention')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers in transformer')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimension')
    parser.add_argument('--num_features', type=int, default=10, help='number of features')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value for RBF')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--cutoff', type=int, default=1, help='Cutoff length for triplets')
    parser.add_argument('--init_seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

    hparams = parser.parse_args()

    pl.seed_everything(hparams.init_seed)

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            "matbench_jdft2d",  # 636
            "matbench_phonons",  # 1,265
            # "matbench_dielectric",  # 4,764
            # "matbench_log_gvrh",  # 10,987
            # "matbench_log_kvrh",  # 10,987
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

            x_input = torch.tensor(get_triplets_with_angles_as_features(train_inputs, max_length))  # 处理输入

            dataset = StructureDataset(x_input, train_outputs)
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=hparams.init_seed)

            train_loader = DataLoader(train_data, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                                      persistent_workers=True)
            val_loader = DataLoader(val_data, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                                    persistent_workers=True)

            model = Predictor(hparams)

            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=300, verbose=True, mode="min")
            checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
            progress_bar = RichProgressBar()
            callbacks=[early_stop_callback,checkpoint_callback,progress_bar]

            trainer = pl.Trainer(max_epochs=1000,
                                 callbacks=callbacks,
                                 log_every_n_steps=50,
                                 accelerator='gpu',
                                 devices=[2])
            trainer.fit(model, train_loader, val_loader)

            # 加载验证损失最小的模型权重
            best_model_path = checkpoint_callback.best_model_path
            best_model = Predictor.load_from_checkpoint(best_model_path, hparams=hparams)

            # 测试
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs = torch.tensor(get_triplets_with_angles_as_features(test_inputs, max_length))
            test_dataset = StructureDataset(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers,
                                     persistent_workers=True)

            predict = trainer.test(best_model, test_loader)

            mae = predict[0]['test_mae']
            mae_list.append(mae)

        visualize_results(mae_list, dataset_name)
