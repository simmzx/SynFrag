import torch
from rdkit import Chem
import dgl
from dgl.data.utils import save_graphs, load_graphs
from torch.utils.data import Dataset


datafile = ""



allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def moltree_to_dgl_graph(batch):
    graph_data_batch = []
    for mol in batch:
        if mol is None:
            continue
        else:
            dgl_graph = smiles_to_dgl_graph(mol.smiles)
            graph_data_batch.append(dgl_graph)
    graph_data_batch = dgl.batch(graph_data_batch)
    return graph_data_batch

def smiles_list_to_dgl_graph_batch(batch):
    graph_data_batch = []
    for s in batch:
        if s is None:
            continue
        else:
            dgl_graph = smiles_to_dgl_graph(s)
            graph_data_batch.append(dgl_graph)
    graph_data_batch = dgl.batch(graph_data_batch)
    return graph_data_batch

def smiles_list_to_dgl(list):
    dgl_graph_list = []
    for s in list:
        if s is None:
            continue
        else:
            dgl_graph = smiles_to_dgl_graph(s)
            dgl_graph_list.append(dgl_graph)

def smiles_to_dgl_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
            atom.GetDegree(),
            atom.GetImplicitValence(),
            atom.GetIsAromatic()
        ]
        atom_features.append(atom_feature)
    atom_features = torch.tensor(atom_features, dtype=torch.float)
    
    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feature = [
            allowable_features['possible_bonds'].index(bond.GetBondType()),
            allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
        ]
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)
    
    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features_list, dtype=torch.float)
    
    g = dgl.graph((edge_index[0], edge_index[1]))
    g.ndata['feat'] = atom_features
    g.edata['feat'] = edge_features
    
    return g

'''
if __name__ == "__main__":
    smiles = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    g = smiles_to_dgl_graph(smiles)
    node_feats = g.ndata['feat']
    edge_feats = g.edata['feat']
    #print("node_feats：", g.ndata['feat'])
    #print("edge_feats：", g.edata['feat'])
    node_feat_size = node_feats.shape[1]
    edge_feat_size = edge_feats.shape[1]
    graph_feat_size = 300
    num_layers = 5
    dropout = 0.2

    model = AttentiveFPGNN(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

    node_representation = model(g, node_feats, edge_feats)
    print(node_representation)
    print(node_representation.shape)
    print(node_representation.dtype)
'''