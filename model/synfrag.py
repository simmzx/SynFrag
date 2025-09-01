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
import tempfile
import time
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
MODEL_PATH = os.path.join(project_root, "checkpoints", "synfrag_default.pth")
BATCH_SIZE = 32
RANDOM_SEED = 31
DEVICE = 0


def parse_args():
    """command line argument parser"""
    parser = argparse.ArgumentParser(
        description='SynFrag: Synthetic Accessibility via Fragment Assembly Generation',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''Examples:
  CSV file mode:
    python synfrag.py --input_file example.csv
    python synfrag.py --input_file molecules.csv --smiles_field "SMILES"
  
  Direct SMILES mode:
    python synfrag.py --smiles "CCO"
    python synfrag.py --smiles "CCO" "CC(=O)O" "c1ccccc1"
    python synfrag.py --smiles "CCO,CC(=O)O,c1ccccc1"'''
    )
    
    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_file', help='Input CSV file with SMILES column')  
    input_group.add_argument('--smiles', nargs='+', help='One or more SMILES strings')
    
    parser.add_argument('--smiles_field', default='smiles', help='SMILES column name for CSV mode (default: smiles)')
    parser.add_argument('--output_file', help='Output CSV file path (optional for SMILES mode)')
    
    return parser.parse_args()


def setup_environment():
    """setup random seed and environment"""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def validate_and_load_data(input_file, smiles_field):
    """validate and load input data from CSV file"""
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


def parse_smiles_input(smiles_input):
    """Parse SMILES input from command line arguments"""
    smiles_list = []
    for item in smiles_input:
        if ',' in item:
            smiles_list.extend([s.strip() for s in item.split(',') if s.strip()])
        else:
            smiles_list.append(item.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_smiles = []
    for smi in smiles_list:
        if smi and smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)
    
    return unique_smiles


def create_temp_csv_for_smiles(smiles_list):
    """为SMILES列表创建临时CSV文件"""
    # 创建临时CSV文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    try:
        with os.fdopen(temp_fd, 'w') as f:
            f.write('smiles,labels\n')  # CSV头部
            for smi in smiles_list:
                f.write(f'"{smi}",0\n')  # 添加 dummy label
        
        return temp_path
    except:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_model():
    """load the pretrained model"""
    model_load_start = time.time()
    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    model = AttentiveFPPredictor(
        node_feat_size=30, edge_feat_size=11, num_layers=4,
        num_timesteps=1, graph_feat_size=300, n_tasks=1, dropout=0
    )
    model.from_pretrained_all(MODEL_PATH)
    model.to(device).eval()
    model_load_time = time.time() - model_load_start
    print(f"✓ Model loaded on {device} (⏱️ {model_load_time:.2f}s)")
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
        
        # fulfill result with scores
        for idx, score in zip(indices, scores):
            results[idx] = score
            
    except Exception as e:
        print(f"Warning: Batch prediction failed - {e}")
    
    return results


def generate_output_path(input_file):
    """Generate output file path based on input file name"""
    input_dir = os.path.dirname(os.path.abspath(input_file))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(input_dir, f"{base_name}_synfrag.csv")


def print_smiles_results(smiles_list, scores):
    """Print results for direct SMILES input"""
    print("\n📊 SynFrag Results:")
    print(f"{'#':<3} {'SMILES':<30} {'SynFrag':<10} {'Status':<10}")
    print("-" * 55)
    
    valid_scores = []
    for i, (smiles, score) in enumerate(zip(smiles_list, scores), 1):
        if score is not None:
            status = "✓"
            valid_scores.append(score)
            print(f"{i:<3} {smiles:<30} {score:<10.3f} {status:<10}")
        else:
            print(f"{i:<3} {smiles:<30} {'N/A':<10} {'✗':<10}")
    
    print(f"\nSummary: {len(valid_scores)}/{len(smiles_list)} successful")
    if valid_scores:
        easy_count = sum(1 for s in valid_scores if s > 0.5)
        print(f"Easy synthesis (>0.5): {easy_count}/{len(valid_scores)} ({easy_count/len(valid_scores)*100:.1f}%)")


def main():
    """main function to run the SynFrag prediction"""
    program_start_time = time.time()
    
    print("🧬 SynFrag: Molecular Synthesis Difficulty Prediction")
    print("=" * 60)
    
    try:
        # 1. argument parsing
        args = parse_args()
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # 2. setup environment
        setup_environment()
        model, device = load_model()

        # 3. determine input mode and process data
        data_prep_start = time.time()
        if args.input_file:
            # CSV mode (原逻辑不变)
            output_file = args.output_file or generate_output_path(args.input_file)
            print(f"📄 CSV File Mode")
            print(f"Input:  {args.input_file}")
            print(f"Output: {output_file}")
            print(f"SMILES column: '{args.smiles_field}'")
            
            df = validate_and_load_data(args.input_file, args.smiles_field)
            dataset = MoleculeDatasetB(args.input_file, smiles_field=args.smiles_field, label_field='labels')
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
        else:
            # SMILES mode (修复后的逻辑)
            print(f"🧪 Direct SMILES Mode")
            smiles_list = parse_smiles_input(args.smiles)
            
            print(f"Input SMILES: {len(smiles_list)} molecules")
            for i, smi in enumerate(smiles_list, 1):
                print(f"  {i}. {smi}")
            
            if args.output_file:
                print(f"Output: {args.output_file}")
            
            # 创建临时CSV文件，使用与CSV模式完全相同的处理方式
            temp_csv_path = create_temp_csv_for_smiles(smiles_list)
            try:
                dataset = MoleculeDatasetB(temp_csv_path, smiles_field='smiles', label_field='labels')
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            except Exception as e:
                if os.path.exists(temp_csv_path):
                    os.unlink(temp_csv_path)
                raise e

        data_prep_time = time.time() - data_prep_start
        print(f"⏱️ Data preparation time: {data_prep_time:.2f}s")

        # 4. predict SynFrag
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        print("\n🔄 Calculating SynFrag...")
        
        prediction_start = time.time()
        all_scores = []
        
        for batch in tqdm(loader, desc="Processing"):
            batch_scores = predict_batch(model, batch, featurizer, device)
            all_scores.extend(batch_scores)

        prediction_time = time.time() - prediction_start
        print(f"⏱️ Prediction time: {prediction_time:.2f}s")

        # 5. handle results based on mode
        results_start = time.time()
        if args.input_file:
            # CSV mode (原逻辑不变)
            df['synfrag'] = all_scores[:len(df)]
            
            valid_scores = [s for s in all_scores[:len(df)] if s is not None]
            success_rate = len(valid_scores) / len(df) * 100
            easy_count = sum(1 for s in valid_scores if s > 0.5)
            
            print(f"\n📊 Results: {len(valid_scores)}/{len(df)} successful ({success_rate:.1f}%)")
            if valid_scores:
                print(f"📈 SynFrag range: {min(valid_scores):.3f} - {max(valid_scores):.3f}")
                print(f"🎯 Easy synthesis (>0.5): {easy_count}/{len(valid_scores)} ({easy_count/len(valid_scores)*100:.1f}%)")
            
            df.to_csv(output_file, index=False)
            print(f"✅ Saved to: {output_file}")
            
        else:
            # SMILES mode (修复后的结果处理)
            try:
                # 清理临时文件
                if os.path.exists(temp_csv_path):
                    os.unlink(temp_csv_path)
                
                # 显示结果
                print_smiles_results(smiles_list, all_scores[:len(smiles_list)])
                
                if args.output_file:
                    df = pd.DataFrame({'smiles': smiles_list, 'synfrag': all_scores[:len(smiles_list)]})
                    df.to_csv(args.output_file, index=False)
                    print(f"✅ Results saved to: {args.output_file}")
                    
            except Exception as e:
                # 确保临时文件被清理
                if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
                    os.unlink(temp_csv_path)
                raise e
        
        results_time = time.time() - results_start
        total_time = time.time() - program_start_time
        
        print(f"\n⏱️  Timing Summary:")
        print(f"   Results processing: {results_time:.2f}s")
        print(f"   Total runtime: {total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()