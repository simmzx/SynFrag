import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import random
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
from scripts.utils.mol.dataset import MoleculeDatasetB
from scripts.utils.mol.model import GNN_graphpred
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
from rdkit import Chem


# global constants
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "checkpoints", "farscore_default.pth")
BATCH_SIZE = 32
RANDOM_SEED = 31
DEVICE = 0


def parse_args():
    """command line argument parser"""
    parser = argparse.ArgumentParser(
        description='FARScore: Molecular Synthetic Accessibility Predictor',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Example: python farscore.py --input_file example.csv'
    )
    parser.add_argument('--input_file', required=True, help='Input CSV file with SMILES')
    parser.add_argument('--smiles_field', default='smiles', help='SMILES column name (default: smiles)')
    return parser.parse_args()


def setup_environment():
    """setup random seed and environment"""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def validate_and_load_data(input_file, smiles_field):
    """validate and load input data"""
    # check file existence and read data
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    df = pd.read_csv(input_file)
    if smiles_field not in df.columns:
        raise ValueError(f"Column '{smiles_field}' not found. Available: {list(df.columns)}")
    
    valid_count = df[smiles_field].notna().sum()
    print(f"✓ Loaded {len(df)} molecules ({valid_count} valid SMILES)")
    return df


def load_model():
    """load the pretrained model"""
    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    model = AttentiveFPPredictor(
        node_feat_size=30, edge_feat_size=11, num_layers=4,
        num_timesteps=1, graph_feat_size=300, n_tasks=1, dropout=0
    )
    model.from_pretrained_all(MODEL_PATH)
    model.to(device).eval()
    print(f"✓ Model loaded on {device}")
    return model, device


def standardize_smiles(smiles):
    """normalize SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None


def predict_batch(model, batch, featurizer, device):
    """predict scores for a batch of SMILES"""
    test_smiles = batch[0]
    results = [None] * len(test_smiles)
    
    # process valid SMILES
    valid_data = []
    for i, smi in enumerate(test_smiles):
        if isinstance(smi, str) and smi.strip():
            std_smi = standardize_smiles(smi.strip())
            if std_smi:
                valid_data.append((i, std_smi))
    
    if not valid_data:
        return results
    
    # featurize valid SMILES and predict
    try:
        indices, smiles_list = zip(*valid_data)
        graphs = featurizer.featurize(smiles_list)
        dgl_graphs = [g.to_dgl_graph(self_loop=True) for g in graphs]
        batch_graph = dgl.batch(dgl_graphs).to(device)
        
        with torch.no_grad():
            pred = model(batch_graph, batch_graph.ndata['x'], batch_graph.edata['edge_attr'])
            scores = (1.0 - pred.detach().cpu().flatten()).tolist()
        
        # fullfill result with scores
        for idx, score in zip(indices, scores):
            results[idx] = score
            
    except Exception as e:
        print(f"Warning: Batch prediction failed - {e}")
    
    return results


def generate_output_path(input_file):
    """Generate output file path based on input file name"""
    input_dir = os.path.dirname(os.path.abspath(input_file))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(input_dir, f"{base_name}_farscore.csv")


def main():
    """main function to run the FARScore prediction"""
    print("🧬 FARScore: Molecular Synthesis Difficulty Prediction")
    print("=" * 60)
    
    try:
        # 1. argument parsing
        args = parse_args()
        output_file = generate_output_path(args.input_file)
        
        print(f"Input:  {args.input_file}")
        print(f"Output: {output_file}")
        print(f"SMILES column: '{args.smiles_field}'")
        
        # 2. setup environment and load data
        setup_environment()
        df = validate_and_load_data(args.input_file, args.smiles_field)
        model, device = load_model()
        
        # 3. prepare dataset and dataloader
        dataset = MoleculeDatasetB(args.input_file, smiles_field=args.smiles_field, label_field='labels')
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        
        # 4. predict FARScore
        print("\n🔄 Calculating FARScore...")
        all_scores = []
        
        for batch in tqdm(loader, desc="Processing"):
            batch_scores = predict_batch(model, batch, featurizer, device)
            all_scores.extend(batch_scores)
        
        # 5. save results
        df['farscore'] = all_scores[:len(df)]  # ensure length matches input
        
        # 6. calculate statistics
        valid_scores = [s for s in all_scores[:len(df)] if s is not None]
        success_rate = len(valid_scores) / len(df) * 100
        easy_count = sum(1 for s in valid_scores if s > 0.5)
        
        print(f"\n📊 Results: {len(valid_scores)}/{len(df)} successful ({success_rate:.1f}%)")
        if valid_scores:
            print(f"📈 FARScore range: {min(valid_scores):.3f} - {max(valid_scores):.3f}")
            print(f"🎯 Easy synthesis (>0.5): {easy_count}/{len(valid_scores)} ({easy_count/len(valid_scores)*100:.1f}%)")
        
        # 7. save to CSV
        df.to_csv(output_file, index=False)
        print(f"✅ Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()