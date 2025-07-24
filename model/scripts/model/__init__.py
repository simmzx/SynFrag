#!/usr/bin/env python3

try:
    from .AttentiveFPGNN import *
    from .utils import *
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")

__all__ = [
    'AttentiveFPGNN',
    'utils'
]

def get_model_info():
    return {
        'architecture': 'AttentiveFP',
        'type': 'Graph Neural Network',
        'task': 'Molecular Synthetic Accessibility Prediction'
    }
