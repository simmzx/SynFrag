#!/usr/bin/env python3
"""
Setup script for SynFrag - Fragment Assembly autoRegressive pretraining based Synthetic Accessibility Predictor
"""

import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    raise RuntimeError("SynFrag requires Python 3.8 or higher")

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "SynFrag: A tool for predicting molecular synthetic accessibility"

def read_requirements():
    requirements = []
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if not line.startswith('# tensorflow'):
                        requirements.append(line)
    
    return requirements

def get_version():
    # 注意：这里的路径可能需要根据您的包结构调整
    version_file = os.path.join(os.path.dirname(__file__), 'model', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "1.0.0"  # 更新为新版本号

DESCRIPTION = "SynFrag: A Synthetic Accessibility Predictor based Fragment Assembly autoRegressive pretrain"
LONG_DESCRIPTION = read_readme()
VERSION = get_version()
AUTHOR = "Xiang Zhang"
AUTHOR_EMAIL = "776206454@qq.com"
URL = "https://github.com/simmzx/SynFrag"  # 更新GitHub链接
LICENSE = "MIT"

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

KEYWORDS = [
    "chemistry",
    "molecular",
    "synthesizability", 
    "synthetic accessibility",  # 添加更清晰的关键词
    "fragment assembly",        # 体现Fragment概念
    "deep learning",
    "graph neural networks",
    "cheminformatics",
    "drug discovery",
    "SMILES"
]

INSTALL_REQUIRES = read_requirements()

EXTRAS_REQUIRE = {
    'tensorflow': ['tensorflow>=2.10.0,<3.0.0'],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.910',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'nbsphinx>=0.8.0',
    ],
}

# 更新命令行入口点，使用新的项目名称
ENTRY_POINTS = {
    'console_scripts': [
        'synfrag=model.synfrag:main',                    # 主要命令
        'synfrag-pretrain=model.synfrag_pretrain:main', # 预训练命令
        'synfrag-finetune=model.synfrag_finetune:main', # 微调命令
        'synfrag-cli=model.cli:main',                    # CLI命令
        
        # 可选：保留旧命令以提供向后兼容性（在过渡期）
        # 'farscore=model.synfrag:main',
        # 'farscore-cli=model.cli:main',
    ],
}

PACKAGE_DATA = {
    'model': [
        'example.csv',
    ],
    '': [
        'checkpoints/*.pth',
        'data/test_dataset/*.csv',
        'data/train_dataset/finetune/*',
        'data/train_dataset/pretrain/*',
        'README.md',
        'LICENSE',
    ]
}

# 更新数据文件路径，可能需要重命名默认模型文件
DATA_FILES = [
    ('checkpoints', ['checkpoints/synfrag_default.pth']),  # 更新模型文件名
    ('data/test_dataset', [
        'data/test_dataset/TS1.csv',
        'data/test_dataset/TS2.csv', 
        'data/test_dataset/TS3.csv',
        'data/test_dataset/TSA.csv',
        'data/test_dataset/TSB.csv'
    ]),
]

setup(
    name="synfrag",  # 更新包名
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    packages=find_packages(include=['model', 'model.*']),
    
    python_requires=">=3.8",
    
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    
    entry_points=ENTRY_POINTS,
    
    include_package_data=True,
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    
    zip_safe=False,
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}/docs",
    },
    
    platforms=["any"],
)