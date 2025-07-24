#!/usr/bin/env python3


try:
    from . import AttentiveFP
    from . import mol
    from . import nn
except ImportError as e:
    print(f"Warning: Could not import utils submodules: {e}")

__all__ = [
    'AttentiveFP',
    'mol', 
    'nn'
]

def get_utils_info():
    return {
        'AttentiveFP': 'AttentiveFP graph neural network implementation',
        'mol': 'Molecular processing and feature extraction tools',
        'nn': 'Neural network creation and training utilities'
    }
