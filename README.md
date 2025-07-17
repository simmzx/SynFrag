![AUR License](https://img.shields.io/aur/license/create-react-app?style=flat) [![我是徽章](https://img.shields.io/badge/simmzx💤%E7%9A%84-GitHub-brightgreen)](https://github.com/simmzx💤/My_Documentation)

# FARScore: Molecular Synthetic Accessibility Predictor
> Fragment Assembly autoRegressive based synthetic accessibility scorer to accelerate drug discovery
## 🎯 What Makes FARScore Different
FARScore revolutionizes synthetic accessibility prediction through **fragment assembly autoregressive pretraining**. Unlike traditional approaches that directly learn synthesis patterns, FARScore first masters molecular construction fundamentals—understanding how molecules are assembled from fragments—then applies this knowledge to predict synthetic accessibility.
### Two-Stage Learning:
* **Stage 1**: Pretrain on 9.2M unlabeled molecules to learn molecular assembly patterns
* **Stage 2**: Finetune on 800K labeled molecules for synthetic accessibility prediction

This mirrors human chemical intuition: experienced chemists understand molecular construction before assessing synthetic difficulty.

## ✨  Key Features
* Easy Integration - Simple CSV input/output format
* Batch Prediction - One-click synthetic accessibility scoring
* High Accuracy - Achieves SOTA performance on multiple test sets with key metrics including accuracy, AUROC and specificity.

## 🚀 Quick Start
### 1. Installation
```python
    # Clone repository
    git clone https://github.com/simmzx/FARScore.git
    cd FARScore

    # Create environment and install dependencies
    conda create -n FARScore python=3.8
    conda activate FARScore
    pip install -r requirements.txt
```
### 2. Prepare Data
Create CSV file with SMILES column:
molecule_id  | smiles|
:---------: | :--------:|
Palbociclib  | CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C |
(+)-Eburnamonine  | [C@]12(C3=C4CCN1CCC[C@@]2(CC(=O)N3C1C4=CC=CC=1)CC)[H] |
### 3. Run Prediction
```python
    python farscore.py \
        --input_file molecules.csv \
        --output_file predictions.csv \
        --input_model_file checkpoints/farscore_default.pth \
```
### 4: View Results
Output file will contain FARScore values:
| molecule_id | smiles  | farscore |
| :------------: |:---------------:|:-----:|
| Palbociclib      | CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C | 0.945263371 |
| (+)-Eburnamonine | [C@]12(C3=C4CCN1CCC[C@@]2(CC(=O)N3C1C4=CC=CC=1)CC)[H]        |    0.02855128 |

**Score Interpretation:**
* Close to 1: Easy to synthesize
- Close to 0: Hard to synthesize
* Threshold 0.5: Binary classification cutoff

## 📖 Advanced Usage
Custom Pretraining and Finetuning task
### Pretrain Model
```python
    python farscore_pretrain.py \
        --dataset data/train_dataset/pretrain/smiles.txt \
        --vocab data/train_dataset/pretrain/fragment.txt \
        --output_model_file checkpoints/pretrained_gnn.pth \
        --epochs 100 \
```
### Finetune Model
```python
    python farscore_finetune.py \
        --input_model_file checkpoints/pretrained_gnn.pth \
        --dataset data/train_dataset/finetune/dataset.csv \
        --epochs 200 \
```
## 🔧 Parameters
| Parameter | Description  | Default |
| :------------ |:---------------| :-----|
| `--input_file`      | Input CSV file path | Required |
| `--output_file`      | Output CSV file path        |   Required |
| `--input_model_file` | Pretrained model path        |    Required |
| `--smiles_field` | SMILES column name        |    "smiles" |
| `--batch_size` | Batch processing size        |    32 |
| `--device` | GPU device number        |    0 |

## 📋 Requirements
* Python 3.8-3.10
* CUDA-enabled GPU (recommended)
* Key dependencies: PyTorch, RDKit, DGL, DeepChem

## 📄 Citation
If you find this work useful for your research, please cite our paper:



## :email: Contact
For questions, please contact: Xiang Zhang (Email: zhangxiang@simm.ac.cn)
______________________________________________________________________________________________________
🌟 **Like this project? Give us a Star**
