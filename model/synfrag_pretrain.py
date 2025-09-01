import deepchem as dc
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
from tqdm import tqdm
import numpy as np
from optparse import OptionParser
from scripts.utils.mol.model import GNN, Motif_Generation
from scripts.utils.mol.dataset import MoleculeDataset, MoleculeDatasetB, moltree_to_graph_data, moltree_to_smiles_list
from scripts.utils.mol.cls import Vocab
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
from scripts.utils.mol.dfs import Motif_Generation_dfs
from scripts.utils.mol.pyg import mol_to_graph_data_obj_simple
from scripts.utils.mol.dgl_g import moltree_to_dgl_graph, smiles_to_dgl_graph
from scripts.utils.mol.AttfpMPNN import AttentiveGRU1, AttentiveGRU2, GetContext, GNNLayer, AttentiveFPGNN
from scripts.utils.mol.AttfpMPNN import AttentiveFPPredictor, AttentiveFPReadout, GlobalPool
import argparse

import rdkit
rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)

# =============================================================================
# hyperparameters and configurations
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.01
LOG_INTERVAL = 50
SAVE_INTERVAL = 10
DEVICE = 0  # GPU device ID，-1 direct to CPU

# model parameters
NODE_FEAT = 30
EDGE_FEAT = 11
GRAPH_FEAT = 400
NUM_LAYERS = 4
DROPOUT = 0.3
HIDDEN_SIZE = 400
LATENT_SIZE = 56

# =============================================================================
# core functions
# =============================================================================

def get_paths():
    """get project paths"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(script_dir) != 'SynFrag' and script_dir != os.path.dirname(script_dir):
        script_dir = os.path.dirname(script_dir)
    
    return {
        'checkpoints': os.path.join(script_dir, "checkpoints"),
        'logs': os.path.join(script_dir, "logs")
    }

def group_node_rep(node_rep, batch_index, batch_size):
    """group node representations by batch index"""
    group, count = [], 0
    for i in range(batch_size):
        num = sum(batch_index == i)
        group.append(node_rep[count:count + num])
        count += num
    return group

def create_models(vocab, device):
    """create GNN and motif generation models"""
    gnn = AttentiveFPGNN(NODE_FEAT, EDGE_FEAT, NUM_LAYERS, GRAPH_FEAT, DROPOUT).to(device)
    motif = Motif_Generation(vocab, HIDDEN_SIZE, LATENT_SIZE, 3, device, "dfs").to(device)
    return gnn, motif

def save_models(gnn, motif, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    if epoch == 'pretrained':
        torch.save(gnn.state_dict(), f"{output_dir}/gnn_pretrained.pth")
        torch.save(motif.state_dict(), f"{output_dir}/motif_pretrained.pth")
    elif epoch == 'interrupted':
        torch.save(gnn.state_dict(), f"{output_dir}/gnn_interrupted.pth")
        torch.save(motif.state_dict(), f"{output_dir}/motif_interrupted.pth")
    else:
        torch.save(gnn.state_dict(), f"{output_dir}/gnn_*epoch*{epoch}.pth")
        torch.save(motif.state_dict(), f"{output_dir}/motif_*epoch*{epoch}.pth")

def train_epoch(gnn, motif, loader, optimizers, device, epoch, log_file):
    """Train for one epoch"""
    gnn.train()
    motif.train()
    opt_gnn, opt_motif = optimizers
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    
    word_acc_sum = topo_acc_sum = loss_sum = 0
    
    for step, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch}')):
        try:
            # prepare batch data
            smiles_list = moltree_to_smiles_list(batch)
            graphs = featurizer.featurize(smiles_list)
            dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
            batch_graph = dgl.batch(dgl_graphs).to(device)
            
            # batch index
            batch_num_nodes = batch_graph.batch_num_nodes()
            batch_index = torch.cat([torch.full((num,), i) for i, num in enumerate(batch_num_nodes)])
            
            # forward propagation
            node_rep = gnn(batch_graph, batch_graph.ndata['x'], batch_graph.edata['edge_attr'])
            node_rep = group_node_rep(node_rep, batch_index, len(batch))
            
            # calculate loss and accuracy
            loss, word_acc, topo_acc = motif(batch, node_rep)
            
            # back propagation
            opt_gnn.zero_grad()
            opt_motif.zero_grad()
            loss.backward()
            opt_gnn.step()
            opt_motif.step()
            
            # loss and accuracy accumulation
            loss_sum += loss.item()
            word_acc_sum += word_acc
            topo_acc_sum += topo_acc
            
            # logging
            if (step + 1) % LOG_INTERVAL == 0:
                avg_loss = loss_sum / LOG_INTERVAL
                avg_word = word_acc_sum / LOG_INTERVAL * 100
                avg_topo = topo_acc_sum / LOG_INTERVAL * 100
                
                log_msg = f"Epoch {epoch} Step {step+1} | Loss: {avg_loss:.4f} | Word: {avg_word:.1f}% | Topo: {avg_topo:.1f}%"
                print(log_msg)
                
                with open(log_file, 'a') as f:
                    f.write(log_msg + '\n')
                
                loss_sum = word_acc_sum = topo_acc_sum = 0
                
        except Exception as e:
            print(f"batch {step} failed: {e}")
            continue

def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='SynFrag pretraining')
    parser.add_argument('--dataset', type=str, required=True, help='SMILES file path')
    parser.add_argument('--vocab', type=str, required=True, help='fragment vocabulary file path')
    parser.add_argument('--device', type=int, default=DEVICE, help='GPU device ID')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data', 'train_dataset', 'pretrain')
    
    # 将简单文件名转换为完整路径
    dataset_path = os.path.join(data_dir, args.dataset)
    vocab_path = os.path.join(data_dir, args.vocab)
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇文件不存在: {vocab_path}")

    
    # setup random seeds and device
    torch.manual_seed(731)
    np.random.seed(731)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(731)
    
    # setup paths
    paths = get_paths()
    os.makedirs(paths['checkpoints'], exist_ok=True)
    os.makedirs(paths['logs'], exist_ok=True)
    log_file = os.path.join(paths['logs'], "pretrain.log")
    
    # check dataset and vocab existence
    assert os.path.exists(dataset_path), f"dataset is not exist: {dataset_path}"
    assert os.path.exists(vocab_path), f"vocab is not exist: {vocab_path}"
    
    # print configurations
    print(f"dataset: {dataset_path}")
    print(f"vocab: {vocab_path}")
    print(f"device: {device}")
    print(f"training epochs: {EPOCHS} | batch size: {BATCH_SIZE} | learning rate: {LR}")
    print("-" * 50)
    
    # load dataset
    dataset = MoleculeDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       collate_fn=lambda x: x, drop_last=True, num_workers=4)
    print(f"size of dataset: {len(dataset)} | number of batches: {len(loader)}")
    
    # create models and optimizers
    vocab = Vocab(vocab_path)
    gnn, motif = create_models(vocab, device)
    opt_gnn = optim.Adam(gnn.parameters(), lr=LR, weight_decay=0.0001)
    opt_motif = optim.Adam(motif.parameters(), lr=1e-3, weight_decay=0)
    
    print(f"GNN parameters: {sum(p.numel() for p in gnn.parameters()):,}")
    print(f"Motif parameters: {sum(p.numel() for p in motif.parameters()):,}")
    print("-" * 50)
    
    # initialize log file
    with open(log_file, 'w') as f:
        f.write(f"SynFrag pretrain\ndataset: {dataset_path}\nvocab: {vocab_path}\ndevice: {device}\n{'-'*30}\n")
    
    # training loop
    print("Training Start!...")
    try:
        for epoch in range(1, EPOCHS + 1):
            train_epoch(gnn, motif, loader, (opt_gnn, opt_motif), device, epoch, log_file)
            
            if epoch % SAVE_INTERVAL == 0:
                save_models(gnn, motif, paths['checkpoints'], epoch)
                print(f"models saved: epoch_{epoch}")
                
    except KeyboardInterrupt:
        print("\ntraining interrupted")
        save_models(gnn, motif, paths['checkpoints'], 'interrupted')
    
    # save pretrained models
    save_models(gnn, motif, paths['checkpoints'], 'pretrained')
    print("Training Finished!")

if __name__ == "__main__":
    main()