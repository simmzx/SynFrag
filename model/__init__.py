#!/usr/bin/env python3
"""
SynFrag: A deep learning tool for predicting molecular synthetic accessibility

Main modules:
-  (synfrag.py) - Core prediction functionality
-  (synfrag_pretrain.py) - Pre-training pipeline
-  (synfrag_finetune.py) - Fine-tuning pipeline  
-  (cli.py) - Command line interface
-  (scripts/model/) - Model architecture components
-  (scripts/utils/) - Utility functions and helpers
"""

# 更新版本信息 - 新项目从1.0.0开始
__version__ = "1.0.0"
__author__ = "Xiang Zhang"
__email__ = "zhangxiang@simm.ac.cn"  # 保持原有的邮箱地址
__description__ = "A deep learning tool based fragment assembly autoregressive pretrain for predicting molecular synthetic accessibility"

# 尝试导入核心功能模块 - 注意这里的模块名都需要更新
try:
    # 从重命名后的核心模块导入主要功能
    from .synfrag import predict_synthesizability, load_model
    from .synfrag_pretrain import pretrain_model
    from .synfrag_finetune import finetune_model

    # 导入命令行接口模块
    from . import cli
    
    # 导入脚本模块
    from . import scripts
    
except ImportError as e:
    # 更新错误提示信息中的项目名称
    print(f"Warning: Could not import some SynFrag modules: {e}")

# 定义包的公共接口 - 这些是用户导入包时可以直接访问的组件
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
    """
    获取当前包的版本号
    
    Returns:
        str: 版本号字符串
    """
    return __version__

def get_package_info():
    """
    获取包的基本信息
    
    Returns:
        dict: 包含包名称、版本、作者和描述的字典
    """
    return {
        'name': 'synfrag',  # 更新包名称
        'version': __version__,
        'author': __author__,
        'description': __description__,
    }