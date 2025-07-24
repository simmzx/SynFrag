#!/usr/bin/env python3

try:
    from .AttfpMPNN import *
    from .assemble import *
    from .basic import *
    from .chemutils import *
    from .cls import *
    from .dataset import *
    from .dfs import *
    from .dgl_g import *
    from .filter_valid_smiles import *
    from .fingerprint import *
    from .model import *
    from .pyg import *
except ImportError as e:
    print(f"Warning: Could not import mol modules: {e}")

__all__ = [
    'AttfpMPNN',
    'assemble',
    'basic',
    'chemutils',
    'cls',
    'dataset',
    'dfs',
    'dgl_g',
    'filter_valid_smiles',
    'fingerprint',
    'model',
    'pyg'
]

def get_mol_tools():
    return {
        'AttfpMPNN': 'AttentiveFP Message Passing Neural Network',
        'assemble': 'Molecular assembly tools',
        'basic': 'Basic molecular operations',
        'chemutils': 'Chemical utility functions',
        'cls': 'Classification tools',
        'dataset': 'Dataset processing utilities',
        'dfs': 'Depth-first search algorithms',
        'dgl_g': 'DGL graph processing',
        'filter_valid_smiles': 'SMILES validation tools',
        'fingerprint': 'Molecular fingerprint generation',
        'model': 'Molecular model utilities',
        'pyg': 'PyTorch Geometric integration'
    }
