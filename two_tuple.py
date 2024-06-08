from __future__ import annotations

import os
import sys
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import argparse
import torch.nn.functional as F
import random
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='liu_attention')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--gamma', type=float, default=1.8, help='Gamma value for RBF')
parser.add_argument('--cutoff', type=int, default=3, help='Cutoff length for triplets')
# parser.add_argument("--max_length", type=int, default=96, help="Maximum length parameter")
# parser.add_argument('--e', type=int, default=0, help='number of node embedding dim')

args = parser.parse_args()


# 获取原子序数
def get_atomic_number(element):
    from pymatgen.core.periodic_table import Element
    return Element(element).Z

# 获取原子距离
def get_atom_distance(structure, i, j):
    return structure.get_distance(i, j)

# 获取原子角度
def get_angle(structure, i, j, k):
    return structure.get_angle(i, j, k)

def get_triplets(structures, max_len):  # 处理成三元组
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
        if len(tensor_data) > max_len:
            tensor_data = tensor_data[:max_len]
        # 将当前结构的三元组数据添加到总列表中
        all_tensor_data.append(tensor_data)

    # 对不足最大长度的子列表进行补充
    for sublist in all_tensor_data:
        while len(sublist) < max_len:
            sublist.append((0, 0, 0.0))

    return all_tensor_data

# 生成距离-距离-角度三元组
def get_triplets_with_angles(structures, max_len):
    all_tensor_data = []
    for structure in structures:
        tensor_data = []
        num_atoms = len(structure)

        if num_atoms == 1:
            tensor_data = [(0.0, 0.0, 0.0)] * max_len
        else:
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    for k in range(j + 1, num_atoms):
                        distance_ij = get_atom_distance(structure, i, j)
                        distance_jk = get_atom_distance(structure, j, k)
                        angle_ijk = get_angle(structure, i, j, k)

                        # 归一化角度
                        normalized_angle = angle_ijk / 180.0

                        # 存储三元组数据
                        triplet_data = (distance_ij, distance_jk, normalized_angle)
                        tensor_data.append(triplet_data)

            tensor_data.sort(key=lambda x: x[2], reverse=True)

            if len(tensor_data) > max_len:
                tensor_data = tensor_data[:max_len]

        # 确保每个结构的tensor_data长度一致
        while len(tensor_data) < max_len:
            tensor_data.append((0.0, 0.0, 0.0))

        all_tensor_data.append(tensor_data)

    return all_tensor_data


class TripletStats: # 统计三元组数量
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

class StructureDataset(Dataset):
    def __init__(self, distance_input, angle_input, target):
        self.distance_input = distance_input
        self.angle_input = angle_input
        self.target = target

    def __len__(self):
        return len(self.distance_input)

    def __getitem__(self, idx):
        distance_input = self.distance_input[idx]
        angle_input = self.angle_input[idx]
        target = self.target.iloc[idx]

        return {'distance_input': distance_input, 'angle_input': angle_input, 'target': target}

# 角度展开处理
class AngleExpansion(nn.Module):
    def __init__(self, num_features: int = 10):
        super(AngleExpansion, self).__init__()
        self.num_features = num_features

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        angle_features = [torch.ones_like(angles), angles]
        for i in range(2, self.num_features):
            Tn = 2 * angles * angle_features[-1] - angle_features[-2]
            angle_features.append(Tn)
        angle_features = torch.stack(angle_features, dim=-1)
        return angle_features

# 距离展开处理
class BondExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionRBF, self).__init__()
        self.num_features = num_features
        self.gamma = gamma

    def forward(self, bond_dist: torch.Tensor) -> torch.Tensor:
        feature_centers = torch.arange(1, self.num_features + 1, device=bond_dist.device).float()
        distance_to_centers = torch.abs(feature_centers - bond_dist.unsqueeze(-1))
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2).squeeze()
        return rbf_values

class BondExpansionLearnable(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionLearnable, self).__init__()
        self.num_features = num_features
        self.centers = nn.Parameter(torch.randn(num_features))
        self.gamma = gamma #nn.Parameter(torch.one(1))

    def __call__(self, bond_dist: torch.Tensor) -> torch.Tensor:
        distance_to_centers = torch.abs(self.centers - bond_dist.unsqueeze(-1))
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2)

        return rbf_values

# 神经网络模型
class AttentionStructureModel(nn.Module):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0.2, num_features=10):
        super(AttentionStructureModel, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=num_features, gamma=args.gamma)
        self.angle_expansion = AngleExpansion(num_features=int(num_features))

        self.gru_distances = nn.GRU(input_size=embedding_dim * 3, hidden_size=hidden_size, batch_first=True, num_layers=3, dropout=dropout)
        self.gru_angles = nn.GRU(input_size=num_features * 3, hidden_size=hidden_size, batch_first=True, num_layers=3, dropout=dropout)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
        self.linear_feed = nn.Linear(hidden_size, 1024)
        self.relu1 = nn.ReLU()
        self.linear_forward = nn.Linear(1024, hidden_size)

        self.self_attention1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size * 2, 32)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32, output_size)
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, output_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.layer_norm4 = nn.LayerNorm(hidden_size)

    def forward(self, distances, angles):
        atom1 = self.embedding(distances[:, :, 0].to(torch.long))
        atom2 = self.embedding(distances[:, :, 1].to(torch.long))
        bond = self.bond_expansion(distances[:, :, 2].float())
        embedded_distances = torch.cat([atom1, atom2, bond], dim=-1)

        bond1 = self.bond_expansion(angles[:, :, 0].float())
        bond2 = self.bond_expansion(angles[:, :, 1].float())
        angle = self.angle_expansion(angles[:, :, 2].float())
        embedded_angles = torch.cat([bond1, bond2, angle], dim=-1)

        gru_output_distances, _ = self.gru_distances(embedded_distances)
        gru_output_angles, _ = self.gru_angles(embedded_angles)

        attention_output_distances, _ = self.self_attention(gru_output_distances, gru_output_distances, gru_output_distances)
        attention_output_angles, _ = self.self_attention(gru_output_angles, gru_output_angles, gru_output_angles)

        attention_output_distances = self.layer_norm1(gru_output_distances + attention_output_distances)
        attention_output_angles = self.layer_norm1(gru_output_angles + attention_output_angles)

        feed_forward_output_distances = self.linear_feed(attention_output_distances)
        feed_forward_output_distances = self.relu1(feed_forward_output_distances)
        feed_forward_output_distances = self.linear_forward(feed_forward_output_distances)

        # feed_forward_output_angles = self.linear_feed(attention_output_angles)
        # feed_forward_output_angles = self.relu1(feed_forward_output_angles)
        # feed_forward_output_angles = self.linear_forward(feed_forward_output_angles)

        attention_output_distances = self.layer_norm2(attention_output_distances + feed_forward_output_distances)
        # attention_output_angles = self.layer_norm2(attention_output_angles + feed_forward_output_angles)

        attention_output_distances, _ = self.self_attention1(attention_output_distances, attention_output_distances, attention_output_distances)
        # attention_output_angles, _ = self.self_attention1(attention_output_angles, attention_output_angles, attention_output_angles)

        combined_output = torch.cat([attention_output_distances[:, -1, :], attention_output_angles[:, -1, :]], dim=-1)

        output = self.final_layers(combined_output)

        output = output.squeeze(1)

        return output


class liu_attention_Lightning(pl.LightningModule):
    def __init__(self):
        super(liu_attention_Lightning, self).__init__()
        self.model = AttentionStructureModel()

    def forward(self, distances, angles):
        distances = distances.float()
        angles = angles.float()
        return self.model(distances, angles)

    def training_step(self, batch, batch_idx):
        distances = batch['distance_input'].float()
        angles = batch['angle_input'].float()
        targets = batch['target'].float()
        outputs = self(distances, angles)
        loss = nn.L1Loss()(outputs, targets)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        distances = batch['distance_input'].float()
        angles = batch['angle_input'].float()
        targets = batch['target'].float()
        outputs = self(distances, angles)
        val_loss = nn.L1Loss()(outputs, targets)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        distances = batch['distance_input'].float()
        angles = batch['angle_input'].float()
        targets = batch['target'].float()
        outputs = self(distances, angles)
        test_loss = F.l1_loss(outputs, targets)
        self.log('test_mae', test_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def prepare_dataloader(train_inputs, train_outputs, max_length, batch_size, num_worker):
    distance_inputs = torch.tensor(get_triplets(train_inputs, max_length), dtype=torch.float32)
    angle_inputs = torch.tensor(get_triplets_with_angles(train_inputs, max_length), dtype=torch.float32)

    dataset = StructureDataset(distance_inputs, angle_inputs, train_outputs)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_worker, persistent_workers=True)

    return train_loader, val_loader

def visualize_results(results_list, mb_dataset_name): # 可视化结果并保存到文件中
    for i, mae in enumerate(results_list):
        print(f"Fold {i} MAE: {mae}")
    average_mae = sum(mae_list) / len(mae_list)
    print(f"Average MAE across all folds: {average_mae}")

    # 写入结果到文件
    with open('add_angle.txt', 'a') as f:
        if f.tell() != 0:
            f.write('\n')
        for fold_num, mae in enumerate(results_list):
            f.write(f"Fold {fold_num}, MAE:{mae}\n")
        f.write(f"{mb_dataset_name}, batch_size:{args.batch_size}, gamma:{args.gamma}, Average MAE: {average_mae}\n")
    results_list.clear()

def set_random_seed(seed): # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    init_seed = 42
    set_random_seed(init_seed)

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            # "matbench_phonons",  # 1,265
            "matbench_jdft2d",  # 636
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
        set_random_seed(init_seed)
        task.load()
        dataset_name = task.dataset_name
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            max_length = TripletStats(train_inputs).get_average() * args.cutoff

            train_loader, val_loader = prepare_dataloader(train_inputs, train_outputs, max_length, args.batch_size,
                                                          num_worker=4)

            lightning_model = liu_attention_Lightning()

            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=300, verbose=True,
                                                mode="min")
            checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

            trainer = pl.Trainer(max_epochs=2000, callbacks=[early_stop_callback, checkpoint_callback],
                                 log_every_n_steps=50)
            trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            best_model_path = checkpoint_callback.best_model_path
            lightning_model = liu_attention_Lightning.load_from_checkpoint(best_model_path)

            torch.save(lightning_model.state_dict(), f'phonons_model_fold_{fold}.pth')

            lightning_model.eval()
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_distance_inputs = torch.tensor(get_triplets(test_inputs, max_length), dtype=torch.float32)
            test_angle_inputs = torch.tensor(get_triplets_with_angles(test_inputs, max_length), dtype=torch.float32)
            test_dataset = StructureDataset(test_distance_inputs, test_angle_inputs, test_outputs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4,
                                     persistent_workers=True)

            predict = trainer.test(model=lightning_model, dataloaders=test_loader)

            mae = predict[0]['test_mae']
            mae_list.append(mae)

        visualize_results(mae_list, dataset_name)









