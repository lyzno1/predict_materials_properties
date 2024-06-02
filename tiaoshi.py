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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv

# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='liu_attention')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value for RBF')
# parser.add_argument('--e', type=int, default=0, help='number of node embedding dim')

args = parser.parse_args()


def get_atomic_number(element):
    return Element(element).Z

def get_atom_distance(structure, index1, index2):
    return structure.get_distance(index1, index2)

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

def get_triplets(structures, max_length):
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
            element_i = str(structure[i].species_string)
            for j in range(i + 1, num_atoms):
                element_j = str(structure[j].species_string)
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

def triplets_to_graph_data(triplets, targets):
    graph_data_list = []
    for triplet, target in zip(triplets, targets):
        nodes = []
        edges = []
        edge_attr = []
        for atom1, atom2, distance in triplet:
            # 添加节点
            if atom1 not in nodes:
                nodes.append(atom1)
            if atom2 not in nodes:
                nodes.append(atom2)
            # 添加边和边属性
            edge_index = [nodes.index(atom1), nodes.index(atom2)]
            edges.append(edge_index)
            edge_attr.append(distance)

        # 转换为张量
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        x = torch.tensor(nodes, dtype=torch.long).view(-1, 1)
        y = torch.tensor([target], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph_data_list.append(data)

    return graph_data_list

# 距离bond expansion处理
class BondExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionRBF, self).__init__()
        self.num_features = num_features
        self.gamma = gamma

    def __call__(self, bond_dist: torch.Tensor) -> torch.Tensor:
        # 生成特征中心张量
        feature_centers = torch.arange(1, self.num_features + 1, device=bond_dist.device).float()

        # 计算每个距离到各个特征中心的欧几里得距离
        distance_to_centers = torch.abs(feature_centers - bond_dist.unsqueeze(-1))

        # 使用高斯径向基函数计算每个距离对应的特征值
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


class StructureDataset(Dataset):
    def __init__(self, structure, graph_data, target):
        self.input_structure = structure
        self.input_graph = graph_data
        self.target = target

    def __len__(self):
        return len(self.input_structure)

    def __getitem__(self, idx):
        structure = self.input_structure[idx]
        graph_data = self.input_graph[idx]
        target = self.target.iloc[idx]

        # Return structure, graph data, and target
        return {'structure': structure, 'graph_data': graph_data, 'target': target}


class Attention_structer_model(nn.Module):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0.2, num_features=10):
        super(Attention_structer_model, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=num_features, gamma=args.gamma)

        self.gru = nn.GRU(input_size=embedding_dim * 3, hidden_size=hidden_size, batch_first=True, num_layers=3,
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

        # 图神经网络部分
        self.gat_conv1 = GATConv(embedding_dim, hidden_size // 2, heads=2)
        self.gat_conv2 = GATConv(hidden_size, hidden_size, heads=2)
        self.gat_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, graph_data):
        # 处理三元组数据
        atom1 = self.embedding(x[:, :, 0].to(torch.long))
        atom2 = self.embedding(x[:, :, 1].to(torch.long))
        bond = self.bond_expansion(x[:, :, 2].float())
        embedded_data = torch.cat([atom1, atom2, bond], dim=-1)

        gru_output, _ = self.gru(embedded_data)
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

        attention_output, _ = self.self_attention1(attention_output, attention_output, attention_output)

        triplet_output = attention_output[:, -1, :]  # 选择最后一个时间步的隐藏状态

        # 处理图数据
        x_graph, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        graph_output = self.gat_conv1(x_graph, edge_index, edge_attr)
        graph_output = F.relu(graph_output)
        graph_output = self.gat_conv2(graph_output, edge_index, edge_attr)
        graph_output = F.relu(graph_output)
        graph_output = torch_geometric.nn.global_mean_pool(graph_output, graph_data.batch)
        graph_output = self.gat_fc(graph_output)

        # 融合三元组和图的特征
        fused_output = triplet_output + graph_output  # 可以选择拼接 torch.cat([triplet_output, graph_output], dim=-1) 再通过全连接层

        # 线性层
        output = self.linear1(self.relu2(self.linear(fused_output)))
        output = output.squeeze(1)

        return output


class liu_attention_Lightning(pl.LightningModule):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0, num_features=10):
        super(liu_attention_Lightning, self).__init__()
        self.model = Attention_structer_model(embedding_dim=embedding_dim, hidden_size=hidden_size,
                                              output_size=output_size, dropout=dropout, num_features=num_features)

    def training_step(self, batch, batch_idx):
        x, label, graph_data = batch['input'], batch['target'].float(), batch['graph_data']
        predict = self.model(x, graph_data)
        loss = F.l1_loss(predict, label)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label, graph_data = batch['input'], batch['target'].float(), batch['graph_data']
        predict = self.model(x, graph_data)
        val_loss = F.l1_loss(predict, label)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, label, graph_data = batch['input'], batch['target'].float(), batch['graph_data']
        predict = self.model(x, graph_data)

        test_loss = F.l1_loss(predict, label)
        self.log('test_mae', test_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


def visualize_results(results_list): # 可视化结果并保存到文件中
    for i, mae in enumerate(results_list):
        print(f"Fold {i} MAE: {mae}")
    average_mae = sum(mae_list) / len(mae_list)
    print(f"Average MAE across all folds: {average_mae}")

    # 写入结果到文件
    with open('results.txt', 'a') as f:
        if f.tell() != 0:
            f.write('\n')
        for fold_num, mae in enumerate(results_list):
            f.write(f"batch_size:{batch_size}, Fold {fold_num}, MAE:{mae}\n")
        f.write(f"batch_size:{batch_size}, gamma:{args.gamma}, Average MAE: {average_mae}\n")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    init_seed = 42
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

            max_length = tripletshu(train_inputs) * 3  # 用于截断/补齐
            x_input = torch.tensor(get_triplets(train_inputs, max_length))  # 处理输入

            # Process graph data for training
            graph_data_train = triplets_to_graph_data(train_inputs, train_outputs)

            dataset = StructureDataset(x_input, train_outputs, graph_data_train)  # Include graph data
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
            batch_size = args.batch_size
            num_worker = 4
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                      persistent_workers=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                                    persistent_workers=True)

            lightning_model = liu_attention_Lightning()

            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=300, verbose=True,
                                                mode="min")
            checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

            trainer = pl.Trainer(max_epochs=1500, callbacks=[early_stop_callback, checkpoint_callback],
                                 log_every_n_steps=50)
            trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # 加载验证损失最小的模型权重
            best_model_path = checkpoint_callback.best_model_path
            lightning_model = liu_attention_Lightning.load_from_checkpoint(best_model_path)

            # 保存最佳模型到 .pth 文件，文件名包含fold编号
            torch.save(lightning_model.state_dict(), f'phonons_model_fold_{fold}.pth')

            # 测试
            lightning_model.eval()
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs = torch.tensor(get_triplets(test_inputs, max_length))

            # Process graph data for testing
            graph_data_test = triplets_to_graph_data(test_inputs, test_outputs)
            test_dataset = StructureDataset(test_inputs, test_outputs, graph_data_test)  # Include graph data
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_worker,
                                     persistent_workers=True)

            predict = trainer.test(model=lightning_model, dataloaders=test_loader)

            mae = predict[0]['test_mae']
            mae_list.append(mae)

        visualize_results(mae_list)








