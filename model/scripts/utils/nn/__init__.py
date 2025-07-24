#!/usr/bin/env python3

try:
    from .create import *
except ImportError as e:
    print(f"Warning: Could not import nn modules: {e}")

__all__ = [
    'create'
]

def get_nn_tools():
    return {
        'create': 'Neural network creation and initialization tools'
    }
