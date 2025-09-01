import argparse
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scripts.utils.mol.dataset import MoleculeDatasetB
from scripts.utils.mol.model import GNN_graphpred
from scripts.utils.mol.pyg import mol_to_graph_data_obj_simple
from rdkit import Chem
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
from scripts.utils.mol.pyg import mol_to_graph_data_obj_simple
from scripts.utils.mol.dgl_g import moltree_to_dgl_graph, smiles_to_dgl_graph
from scripts.utils.mol.AttfpMPNN import AttentiveGRU1, AttentiveGRU2, GetContext, GNNLayer, AttentiveFPGNN
from scripts.utils.mol.AttfpMPNN import AttentiveFPPredictor, AttentiveFPReadout, GlobalPool
import deepchem as dc
import dgl
import torch.nn.functional as F
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models import AttentiveFP, AttentiveFPModel
from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
import os

import rdkit
rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)

# =============================================================================
# hyperparameters and configurations
# =============================================================================
EPOCHS = 200
BATCH_SIZE = 16
GNN_LR = 0.01
POOL_LR = 0.001
GNN_DECAY = 0
POOL_DECAY = 0.001
LOG_INTERVAL = 2000
PATIENCE = 5
DEVICE = 0

# model parameters
NODE_FEAT = 30
EDGE_FEAT = 11
NUM_LAYERS = 4
NUM_TIMESTEPS = 1
GRAPH_FEAT = 400
N_TASKS = 1
DROPOUT = 0.5

# =============================================================================
# core functions
# =============================================================================

def get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(script_dir) != 'SynFrag' and script_dir != os.path.dirname(script_dir):
        script_dir = os.path.dirname(script_dir)
    return {
        'checkpoints': script_dir + "/checkpoints",
        'logs': script_dir + "/logs",
        'data_train': script_dir + "/data/train_dataset/finetune",
        'data_test': script_dir + "/data/test_dataset"
    }

def resolve_path(file_input, default_dir):
    if os.path.isabs(file_input) and os.path.exists(file_input): return file_input
    if os.path.exists(file_input): return os.path.abspath(file_input)
    default_path = os.path.join(default_dir, file_input)
    if os.path.exists(default_path): return default_path
    raise FileNotFoundError(f"file is not exist: {file_input}\ntry the directory: {file_input}, {default_path}")

def create_model(device, model_input, checkpoints_dir):
    model = AttentiveFPPredictor(NODE_FEAT, EDGE_FEAT, NUM_LAYERS, NUM_TIMESTEPS, GRAPH_FEAT, N_TASKS, DROPOUT).to(device)
    try:
        model_path = resolve_path(model_input, checkpoints_dir)
        model.gnn.from_pretrained(model_path)
        print(f"load model successfully: {model_path}")
    except:
        print(f"pretrained model not found: {model_input}, using default parameters")
    return model

def evaluate(model, loader, device, name):
    if not loader: return 0.0
    model.eval()
    y_list, pred_list = [], []
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'testing{name}'):
            graphs = featurizer.featurize(batch[0])
            dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
            batch_graph = dgl.batch(dgl_graphs).to(device)
            pred = model(batch_graph, batch_graph.ndata['x'], batch_graph.edata['edge_attr'])
            y = batch[1].view(pred.shape).to(torch.float).to(device)
            y_list.extend(y.flatten().tolist())
            pred_list.extend(pred.detach().cpu().flatten().tolist())
    return roc_auc_score(y_list, pred_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_file', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(731); np.random.seed(731)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(731)
    
    paths = get_paths()
    os.makedirs(paths['checkpoints'], exist_ok=True)
    os.makedirs(paths['logs'], exist_ok=True)
    log_file = paths['logs'] + "/finetune.log"
    
    print(f"dataset: {args.dataset} | model: {args.input_model_file} | device: {device}")
    
    # load dataset
    train_path = resolve_path(args.dataset, paths['data_train'])
    ts2_path = paths['data_test'] + "/TS2.csv"
    ts3_path = paths['data_test'] + "/TS3.csv"
    
    train_dataset = MoleculeDatasetB(train_path, 'smiles', 'labels')
    test_dataset_ts2 = MoleculeDatasetB(ts2_path, 'smiles', 'labels') if os.path.exists(ts2_path) else None
    test_dataset_ts3 = MoleculeDatasetB(ts3_path, 'smiles', 'labels') if os.path.exists(ts3_path) else None
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    test_loader_ts2 = DataLoader(test_dataset_ts2, BATCH_SIZE, shuffle=False, num_workers=4) if test_dataset_ts2 else None
    test_loader_ts3 = DataLoader(test_dataset_ts3, BATCH_SIZE, shuffle=False, num_workers=4) if test_dataset_ts3 else None
    
    print(f"train dataset: {len(train_dataset)} | number of batches: {len(train_loader)}")
    if test_dataset_ts2: print(f"TS2: {len(test_dataset_ts2)}")
    if test_dataset_ts3: print(f"TS3: {len(test_dataset_ts3)}")
    
    # create model and optimizer
    model = create_model(device, args.input_model_file, paths['checkpoints'])
    optimizer = optim.Adam([
        {'params': model.gnn.parameters(), 'lr': GNN_LR, 'weight_decay': GNN_DECAY},
        {'params': model.readout.parameters(), 'lr': POOL_LR, 'weight_decay': POOL_DECAY},
        {'params': model.predict.parameters(), 'lr': POOL_LR, 'weight_decay': POOL_DECAY}
    ])
    loss_fn = nn.MSELoss()
    
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    with open(log_file, 'w') as f:
        f.write(f"SynFrag finetune | dataset: {args.dataset} | imput model: {args.input_model_file} | device: {device}\n")
    
    # training loop
    best_auroc_ts2 = best_auroc_ts3 = patience_counter = 0
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = num_batches = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            graphs = featurizer.featurize(batch[0])
            dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
            batch_graph = dgl.batch(dgl_graphs).to(device)
            
            preds = model(batch_graph, batch_graph.ndata['x'], batch_graph.edata['edge_attr'])
            labels = batch[1].view(preds.shape).to(torch.float).to(device)
            loss = loss_fn(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (step + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / num_batches
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch} Step {step+1} Loss: {avg_loss:.4f}\n")
        
        # evaluate on test sets
        auroc_ts2 = evaluate(model, test_loader_ts2, device, "TS2")
        auroc_ts3 = evaluate(model, test_loader_ts3, device, "TS3")
        
        # save models
        torch.save(model.state_dict(), paths['checkpoints'] + "/synfrag.pth")
        torch.save(model.gnn.state_dict(), paths['checkpoints'] + "/finetune_gnn.pth")
        
        with open(paths['checkpoints'] + "/auroc.txt", 'a') as f:
            f.write(f"Epoch {epoch} | TS2: {auroc_ts2:.4f} | TS3: {auroc_ts3:.4f}\n")
        
        # logging results
        if test_loader_ts2 and test_loader_ts3:
            result = f"Epoch {epoch} | Loss: {total_loss/num_batches:.4f} | TS2: {auroc_ts2:.4f} | TS3: {auroc_ts3:.4f}"
        else:
            result = f"Epoch {epoch} | Loss: {total_loss/num_batches:.4f}"
        print(result)
        
        with open(log_file, 'a') as f:
            f.write(result + '\n')
        
        # earling stopping
        if test_loader_ts2 and test_loader_ts3:
            if auroc_ts2 > best_auroc_ts2 and auroc_ts3 > best_auroc_ts3:
                best_auroc_ts2, best_auroc_ts3, patience_counter = auroc_ts2, auroc_ts3, 0
                print(f"The best: TS2={auroc_ts2:.4f}, TS3={auroc_ts3:.4f}")
            elif auroc_ts2 < best_auroc_ts2 and auroc_ts3 < best_auroc_ts3:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("early stopping!")
                    break
            else:
                patience_counter = 0
    
    print("Finetuning Finished!")

if __name__ == "__main__":
    main()