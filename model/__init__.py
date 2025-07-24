#!/usr/bin/env python3
"""
FARScore: A deep learning tool for predicting molecular synthetic accessibility


-  (farscore.py)
-  (farscore_pretrain.py)
-  (farscore_finetune.py)
-  (cli.py)
-  (scripts/model/)
-  (scripts/utils/)
"""

__version__ = "1.0.1"
__author__ = "Xiang Zhang"
__email__ = "zhangxiang@simm.ac.cn"
__description__ = "A deep learning tool based fragment assembly autoregressive pretrain for predicting molecular synthetic accessibility"

try:
    from .farscore import predict_synthesizability, load_model
    from .farscore_pretrain import pretrain_model
    from .farscore_finetune import finetune_model

    from . import cli
    
    from . import scripts
    
except ImportError as e:
    print(f"Warning: Could not import some FARScore modules: {e}")

__all__ = [
    '__version__',
    'predict_synthesizability',
    'load_model',
    'pretrain_model',
    'finetune_model',
    'cli',
    'scripts',
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
