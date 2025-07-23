import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
from scripts.utils.mol.cls import MolTree
from scripts.utils.mol.pyg import mol_to_graph_data_obj_simple
from scripts.utils.mol.dgl_g import smiles_list_to_dgl
from torch_geometric.data import Batch
import pickle
import os
import csv


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        file_extension = os.path.splitext(data_file)[1]  
        if file_extension == '.txt':
            with open(data_file) as f:
                self.data = [line.strip("\r\n ").split()[0] for line in f]
        elif file_extension == '.csv':
            with open(data_file, newline='') as f:
                reader = csv.DictReader(f)  
                self.data = [row['smiles'] for row in reader]  
        else:
            raise ValueError("Unsupported file format. Please provide a .txt or .csv file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            smiles = self.data[idx]
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            mol_tree.assemble()
            return mol_tree
        except:
            return None


class MoleculeDatasetB(Dataset):
    def __init__(self,
                 data_file,
                 smiles_field,
                 label_field):
        df = pd.read_csv(data_file)
        smiles_list = df[smiles_field].to_list()
        label_list = df[label_field].to_list()
        self.smiles = smiles_list
        self.labels = label_list

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


def moltree_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        if mol is None:
            continue
        else:
            graph_data_batch.append(mol_to_graph_data_obj_simple(mol.mol))
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch

def moltree_to_smiles_list(batch):
    smiles_list = []
    for mol in batch:
        if mol is None:
            continue
        else:
            smiles_list.append(mol.smiles)

    return smiles_list

def dgl_graph_generator(data_file, dgl_file):
    with open(data_file) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]
        dgl_graph_list = smiles_list_to_dgl(data)
        print(f"dgl_graph_list_len: {len(dgl_graph_list)}")
        with open(dgl_file, 'wb') as f:
            pickle.dump(dgl_graph_list, f)
        print("generated dgl_g successfully!")


def mol_tree_generator(data_file, moltree_file):
    mol_tree_list = []
    with open(data_file) as f:
        smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        for s in smiles_list:
            mol_tree = MolTree(s)
            mol_tree.recover()
            mol_tree.assemble()
            mol_tree_list.append(mol_tree)
        print(f"mol_tree_list_len: {len(mol_tree_list)}")
        with open(moltree_file, 'wb') as f:
            pickle.dump(mol_tree_list, f)
        print("generated mol_tree successfully!")

