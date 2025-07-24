#!/usr/bin/env python3
"""
FARScore: A deep learning tool for predicting molecular synthetic accessibility

This package provides tools for:
- Predicting molecular synthetic accessibility from SMILES strings
- Training and fine-tuning models on custom datasets
- Processing molecular data for machine learning
"""

__version__ = "1.0.0"
__author__ = "Xiang Zhang"
__email__ = "776206454@qq.com"
__description__ = "FARScore' is a synthetic accessibility predictor as Python module that allows you to calculating molecules' synthetic accessibility"

try:
    from .farscore import predict_synthesizability, load_model
    from .farscore_pretrain import pretrain_model
    from .farscore_finetune import finetune_model
except ImportError:
    pass

__all__ = [
    '__version__',
    'predict_synthesizability',
    'load_model',
    'pretrain_model',
    'finetune_model',
]

def get_version():
    return __version__

def get_package_info():
    return {
        'name': 'farscore',
        'version': __version__,
        'author': __author__,
        'description': __description__,
    }