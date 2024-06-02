import torch
from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
from matbench.bench import MatbenchBenchmark
from matbench.task import MatbenchTask
from pymatgen.analysis.local_env import CrystalNN
from transformers import BertTokenizer
import numpy as np

if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获得matbench的数据集
mb = MatbenchBenchmark(autoload=False)
# 打开其中的一个任务
task = MatbenchTask("matbench_phonons")
fold_number = 0
train_data_matbench = task.get_train_and_val_data(fold_number)
test_data_matbench = task.get_test_data(fold_number, include_target=True)
X, y = train_data_matbench
X1, y1 = test_data_matbench
print(y)

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
    atom_distance = site_i.distance(site_j)
    return atom_distance


def get_triplets(structures):
    all_tensor_data = []  # 存储所有结构的三元组数据
    for structure in structures:
        tensor_data = []
        num_atoms = len(structure)

        if num_atoms == 1:
            lattice = structure.lattice
            atom_symbol = structure[0].species_string
            triplet_data = (atom_symbol, atom_symbol, lattice.a)
            tensor_data.append(triplet_data)
            triplet_data = (atom_symbol, atom_symbol, lattice.b)
            tensor_data.append(triplet_data)
            triplet_data = (atom_symbol, atom_symbol, lattice.c)
            tensor_data.append(triplet_data)
            all_tensor_data.append(tensor_data)
            continue

        for i in range(num_atoms):
            element_i = structure[i].species_string
            for j in range(i + 1, num_atoms):
                element_j = structure[j].species_string
                atom_distance = get_atom_distance(structure, i, j)

                # 存储原始的三元组数据
                triplet_data = (element_i, element_j, atom_distance)
                tensor_data.append(triplet_data)

        # 对三元组列表按照最后一个元素（距离信息）进行升序排序
        tensor_data.sort(key=lambda x: x[2], reverse=False)

        # 将当前结构的三元组数据添加到总列表中
        all_tensor_data.append(tensor_data)

    max_length = max(len(sublist) for sublist in all_tensor_data)
    print(max_length)
    # # 寻找第一个元素为零的三元组并返回索引
    # for idx, tensor_data in enumerate(all_tensor_data):
    #     if tensor_data and tensor_data[0][0] == 0:
    #         print(f"Fault in structure at index {idx}, Triplet: {tensor_data}")

    # 对不足最大长度的子列表进行补充
    for sublist in all_tensor_data:
        while len(sublist) < max_length:
            sublist.append((0, 0, 0.0))

    return all_tensor_data

crystals = get_triplets(X)
crystals = torch.tensor(crystals)
crystals = crystals.to(device)

X1 = get_triplets(X1)
X1 = torch.tensor(X1)
X1 = X1.to(device)


# 假设你的晶体数据集保存在crystals中，是一个三层的嵌套列表
# crystals 是一个列表，每个元素表示一个晶体
# 每个晶体又是一个列表，每个元素是一个三元组 (atomic_symbol1, atomic_symbol2, distance)

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建一个列表来保存所有晶体的所有三元组的文本描述
all_text_sequences = []

# 遍历整个数据集
for crystal in crystals:
    crystal_text_sequences = []
    # 遍历晶体的每个三元组
    for triple in crystal:
        atomic_symbol1 = "Element" + triple[0]
        atomic_symbol2 = "Element" + triple[1]
        distance = str(triple[2])
        text_sequence = f"{atomic_symbol1} {atomic_symbol2} distance {distance}"  # 创建描述性文本
        crystal_text_sequences.append(text_sequence)

    # 将当前晶体的所有三元组的文本描述保存到整个数据集的列表中
    all_text_sequences.append(crystal_text_sequences)

# 使用BERT tokenizer将所有文本序列转换为模型输入
inputs = tokenizer(all_text_sequences, return_tensors="pt", padding=True, truncation=True)

# 将模型输入传递给BERT模型进行推理
model = BertModel.from_pretrained('bert-base-uncased')
outputs = model(**inputs)

# 处理BERT模型的输出...
