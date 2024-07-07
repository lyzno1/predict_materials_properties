from __future__ import annotations

import os
import sys
import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import argparse
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import PowerTransformer
from transformers import BertTokenizer, BertModel

from TripletStats import TripletStats
from pytorch_lightning import seed_everything

def get_atomic_number(element):
    from pymatgen.core.periodic_table import Element
    return Element(element).Z

def get_atom_distance(structure, i, j):
    return structure.get_distance(i, j)

def get_bert_sequences(structures, max_len, tokenizer):
    all_tensor_data = []  # 存储所有结构的BERT输入序列
    for structure in structures:
        sequence_data = []
        num_atoms = len(structure)

        if num_atoms == 1:
            lattice = structure.lattice
            atom_symbol = structure[0].species_string
            atomic_number = get_atomic_number(atom_symbol)
            triplet_text_a = f'{atomic_number}-{atomic_number}-{lattice.a:.2f}'
            triplet_text_b = f'{atomic_number}-{atomic_number}-{lattice.b:.2f}'
            triplet_text_c = f'{atomic_number}-{atomic_number}-{lattice.c:.2f}'
            sequence_data.append(triplet_text_a)
            sequence_data.append(triplet_text_b)
            sequence_data.append(triplet_text_c)
            all_tensor_data.append(sequence_data)
            continue

        for i in range(num_atoms):
            element_i = structure[i].species_string
            for j in range(i + 1, num_atoms):
                element_j = structure[j].species_string
                distance = get_atom_distance(structure, i, j)

                atomic_number_i = get_atomic_number(element_i)
                atomic_number_j = get_atomic_number(element_j)
                triplet_text = f'{atomic_number_i}-{atomic_number_j}-{distance:.2f}'
                sequence_data.append(triplet_text)

        sequence_data.sort(key=lambda x: float(x.split('-')[2]), reverse=False)

        if len(sequence_data) > max_len:
            sequence_data = sequence_data[:max_len]
        all_tensor_data.append(sequence_data)

    bert_input_ids = [tokenizer.encode(' '.join(sequence), padding='max_length', max_length=max_len, truncation=True) for sequence in all_tensor_data]

    return bert_input_ids


class StructureDataset(Dataset):
    def __init__(self, structures, targets, tokenizer, max_len):
        self.structures = structures
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.structures)

    def bert_sequence_function(self, structure):
        # Assuming structure is already in a tensor format
        # Process tensor to extract triplets
        triplets = []
        for i in range(1, len(structure) - 2, 4):  # Adjust this indexing based on your tensor structure
            if i + 2 < len(structure):
                triplet = (structure[i], structure[i+1], structure[i+2].item())
                triplets.append(triplet)
            else:
                break  # If there are not enough elements to form a triplet, break the loop
        return triplets

    def __getitem__(self, idx):
        structure = self.structures[idx]
        targets = self.targets[idx].squeeze(-1)

        # 直接将整个结构信息转换为文本表示
        structure_str = str(structure)
        encoding = self.tokenizer.encode_plus(
            structure_str,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(targets, dtype=torch.float)
        }


class BertAttentionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=768, output_size=1, dropout=0.2):
        super(BertAttentionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0, :]  # [CLS] token的输出

        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        output = self.fc3(x)

        return output.squeeze(1)


class L_BERT(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertAttentionModel(bert_model_name=hparams.model_name, hidden_size=hparams.hidden_size,
                                        output_size=hparams.output_size, dropout=hparams.dropout)
        self.lr = hparams.lr
        self.pt = pt
        # self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

    def training_step(self, batch, batch_idx):
        ids, mask, targets = batch['input_ids'], batch['attention_mask'], batch['targets'].float()
        predict = self.model(ids, mask)
        loss = F.l1_loss(predict, targets)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, targets = batch['input_ids'], batch['attention_mask'], batch['targets'].float()
        predict = self.model(ids, mask)
        loss = F.l1_loss(predict, targets)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        ids, mask, targets = batch['input_ids'], batch['attention_mask'], batch['targets'].float()
        predict = self.model(ids, mask)

        original_predict = self.pt.inverse_transform(predict.cpu().numpy().reshape(-1,1))
        test_loss = F.l1_loss(torch.from_numpy(original_predict).to('cuda'), targets.unsqueeze(-1))
        self.log('test_mae', test_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='liu_attention')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model_name', type=str, default='./bert-base-uncased-model', help='model name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
    parser.add_argument('--output_size', type=int, default=1, help='output size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--max_epochs', type=int, default=300, help='max epochs for training')
    # parser.add_argument('--gamma', type=float, default=1.0, help='Gamma value for RBF')
    # parser.add_argument('--cutoff', type=int, default=1, help='Cutoff length for triplets')

    args = parser.parse_args()


    seed_everything(args.seed)

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            "matbench_jdft2d",  # 636
            # "matbench_phonons",  # 1,265
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
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-tokenizer')

            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            max_length = TripletStats(train_inputs).get_median()
            x_input = torch.tensor(get_bert_sequences(train_inputs, max_length, tokenizer))

            pt = PowerTransformer(method='box-cox')
            transformed_train_outputs = pt.fit_transform(train_outputs.values.reshape(-1, 1))
            dataset = StructureDataset(x_input, transformed_train_outputs, tokenizer, max_length)

            train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)
            train_data = Subset(dataset, train_indices)
            val_data = Subset(dataset, val_indices)

            num_worker = 4
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_worker)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=num_worker)

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                filename='{epoch}-{val_loss:.4f}',
                save_top_k=1,
                mode='min'
            )

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=args.patience,
                mode='min',
                verbose=True,
                min_delta=0
            )

            model = L_BERT(hparams=args)

            trainer = L.Trainer(
                max_epochs=args.max_epochs,
                accelerator='auto',
                callbacks=[checkpoint_callback, early_stopping],
            )

            trainer.fit(model, train_loader, val_loader)

            best_model_path = checkpoint_callback.best_model_path
            model = L_BERT.load_from_checkpoint(best_model_path, hparams=args)

            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            x_test = torch.tensor(get_bert_sequences(test_inputs, max_length, tokenizer))

            test_dataset = StructureDataset(x_test, test_outputs, tokenizer, max_length)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_worker)

            mae = trainer.test(model, test_loader, verbose=False)
            mae_list.append(mae[0]['test_mae'])
            print(f'{dataset_name} - Fold {fold} MAE: {mae[0]["test_mae"]:.4f}')

        print(f'Overall MAE: {np.mean(mae_list):.4f}')
