from scripts.utils.mol.dataset import MoleculeDataset
from torch.utils.data import Dataset
import os
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
from scripts.utils.mol.cls import MolTree
from scripts.utils.mol.pyg import mol_to_graph_data_obj_simple
from torch_geometric.data import Batch

def filter_valid_smiles(dataset, filter_smiles):
    dataset = MoleculeDataset(dataset)
    valid_smiles = []

    for idx in range(len(dataset)):
        mol_tree = dataset[idx]
        if mol_tree is not None:
            valid_smiles.append(dataset.data[idx])

    with open(filter_smiles, 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + '\n')

if __name__ == "__main__":
    dataset = ''
    filter_smiles = ''
    filter_valid_smiles(dataset, filter_smiles)
    print(f"Filtered SMILES have been written to {filter_smiles}")