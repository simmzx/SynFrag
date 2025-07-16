FARScore: Molecular Synthetic Accessibility Score based Fragment Assembly autoRegressive Pretraining
Advancing molecular synthesizability prediction through fragment assembly autoregressive pretraining

🚀 Overview
FARScore revolutionizes molecular synthetic accessibility prediction by leveraging the power of fragment assembly autoregressive pretraining. Unlike traditional approaches that directly learn synthetic accessibility patterns, FARScore first masters the fundamental art of molecular construction—understanding how molecules are assembled from fragments—before applying this knowledge to predict synthetic feasibility.

🧠 Core Innovation
Fragment Assembly autoRegressive Learning
The key insight behind FARScore lies in its two-stage learning paradigm:
Stage 1: Fragment Assembly Pretraining
Trains on 9.2 million unlabeled commercially available molecules
Learns intrinsic molecular assembly patterns through autoregressive fragment reconstruction
Captures deep structural and chemical knowledge about how molecules are constructed
Builds a foundation of molecular "construction grammar"
Stage 2: Synthesizability Fine-tuning
Fine-tunes on 800K labeled molecules (balanced dataset: 50% easy, 50% hard to synthesize)
Applies fragment assembly knowledge to synthesizability prediction
Transforms structural understanding into practical synthetic feasibility assessment
This approach mirrors human chemical intuition: experienced chemists first understand how molecules are built before they can assess synthetic difficulty.

🎯 State-of-the-Art Performance
FARScore achieves SOTA performance across comprehensive benchmark evaluations:
Standard Benchmarks:
TS1, TS2, TS3: Superior performance in AUROC, accuracy, and specificity, etc.
Complementary Validation:
TSA (Literature-based): Excels on molecules from actual synthetic routes
TSB (Generated molecules): Demonstrates strong generalization to novel molecular structures

🔧 Model Architecture
Built on AttentiveFP (Attentive Fingerprinting), FARScore combines:
Graph neural network representations for molecular structure
Attention mechanisms for fragment-level understanding
Autoregressive learning for sequential assembly patterns

Output: Continuous values (0-1) with threshold 0.5
Close to 0: Difficult to synthesize
Close to 1: Easy to synthesize

🛠️ Installation Options
System Requirements
Before installing FARScore, ensure your system meets these requirements:
Python: 3.8 or higher
CUDA: Recommended for GPU acceleration (optional but strongly recommended)
Memory: At least 8GB RAM for basic usage, 16GB+ recommended for large datasets
Storage: At least 5GB free space for dependencies and model checkpoints

Core Dependencies
The FARScore system relies on several key scientific computing and deep learning libraries:
Deep Learning Framework
PyTorch 1.12.0 or higher with CUDA support
PyTorch Geometric for graph neural networks
DGL (Deep Graph Library) for graph processing

Chemistry and Molecular Informatics
RDKit for molecular representation and manipulation
DeepChem for chemical featurization and modeling

Scientific Computing
NumPy, Pandas, SciPy for numerical operations
Scikit-learn for machine learning utilities

Installation Steps
Option 1: Complete Installation via pip
This method installs all dependencies through pip, which works on most systems:
# Clone the repository (assuming you have access)
git clone https://github.com/your-repo/farscore.git
cd farscore

# Install all dependencies
pip install -r requirements.txt
