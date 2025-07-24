#!/usr/bin/env python3

try:
    from . import model
    from . import utils
except ImportError as e:
    print(f"Warning: Could not import scripts submodules: {e}")

__all__ = [
    'model',
    'utils'
]

def get_scripts_info():
    return {
        'model': 'Core model implementations (AttentiveFP, etc.)',
        'utils': 'Utility functions for molecular processing and neural networks'
    }
