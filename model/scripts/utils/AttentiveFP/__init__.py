#!/usr/bin/env python3

try:
    from .AttentiveLayers import *
    from .Featurizer import *
    from .getFeatures import *
except ImportError as e:
    print(f"Warning: Could not import AttentiveFP modules: {e}")

__all__ = [
    'AttentiveLayers',
    'Featurizer',
    'getFeatures'
]

def get_attentivefp_info():
    return {
        'model_type': 'Graph Neural Network',
        'architecture': 'AttentiveFP',
        'components': ['AttentiveLayers', 'Featurizer', 'getFeatures']
    }
