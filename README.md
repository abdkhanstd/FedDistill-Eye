# Paper
**"Beyond parameter sharing: a feature distillation framework for privacy-preserving
 ophthalmic disease classification"**

## FedDistill-Eye: Privacy-Preserving Ophthalmic Disease Classification

This repository contains the implementation of **FedDistill-Eye**, a federated learning framework with knowledge distillation for privacy-preserving ophthalmic disease classification across multiple hospitals.

## Key Features
- **Privacy-Preserving**: No raw data sharing between hospitals
- **Knowledge Distillation**: Transfers knowledge through learned features and soft predictions
- **Multi-Domain**: Supports multiple ophthalmic disease datasets
- **Vision Transformer**: Uses ViT-Large backbone with custom classification head
- **Comprehensive Evaluation**: Extensive testing on multiple benchmark datasets

## Project Structure

```
FedDistill-Eye/
├── configs/
│   └── config.py                 # Configuration parameters
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # Dataset handling and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py             # Model architecture and utilities
│   ├── training/
│   │   ├── __init__.py
│   │   └── federated.py         # Federated learning logic
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py           # Evaluation metrics and testing
├── scripts/
│   └── download_datasets.sh     # Dataset download script
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FedDistill-Eye.git
cd FedDistill-Eye

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Download Datasets
The datasets can be downloaded from the RETFound repository benchmarks:
- **Benchmark & Dataset Links**: [RETFound_MAE Benchmark](https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md)

#### Supported Datasets
- **JSIEC** (Primary taxonomy reference)
- **APTOS2019** (Diabetic Retinopathy)
- **MESSIDOR2** (Diabetic Retinopathy)
- **IDRiD** (Diabetic Retinopathy)
- **OCTID** (OCT Images)
- **PAPILA** (Glaucoma)
- **Glaucoma_fundus** (Glaucoma)
- **Retina** (Multi-disease)

#### Dataset Organization
Organize your datasets in the following structure:

```
datasets/
├── JSIEC/
│   ├── train/
│   │   ├── 0.0.Normal/
│   │   ├── 0.3.DR1/
│   │   ├── 1.0.DR2/
│   │   └── ...
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
├── APTOS2019/
│   ├── train/
│   │   ├── anodr/
│   │   ├── bmilddr/
│   │   └── ...
│   ├── val/
│   └── test/
└── (other datasets with similar structure)
```

**Important**: Each dataset should have `train/`, `val/`, and `test/` splits with class folders containing images.

### 3. Pretrained Models

#### Self-Supervised Pretrained Model
Download the self-supervised pretrained model (`best_model.pth`) from:
- **Repository**: [ATLASS](https://github.com/abdkhanstd/ATLASS)

#### Paper's Pretrained Weights
Download the pretrained weights used in the paper from:
- **OneDrive**: [Pretrained Weights](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/EsMao40xHZRPjDaK_KjR1A0BbTq9_8bTVs8LIM642cGeyw?e=E6QDtQ)

Place the downloaded `best_model.pth` in the root directory of the project.

## Usage

### Training

#### Full Federated Training
```bash
# Run complete federated learning training
python train.py

# Training will create checkpoints in ./checkpoints_federated/
# Best model will be saved as best_federated_model.pth
```

#### Training Configuration
Modify `configs/config.py` to adjust:
- Number of federated rounds (`EPOCHS = 50`)
- Local training epochs (`LOCAL_EPOCHS = 5`)
- Batch size (`BATCH_SIZE = 64`)
- Learning rate (`LEARNING_RATE = 5e-5`)
- Hospital dataset assignments (`HOSPITAL_DATASETS`)

### Evaluation

#### Test on All Datasets
```bash
# Evaluate trained model on all datasets
python train.py --test-only --model-path best_federated_model.pth
```

#### Test on Specific Dataset
```bash
# Test on a specific dataset
python train.py --test-only --dataset JSIEC --model-path best_federated_model.pth
```

#### Show Dataset Statistics
```bash
# Display dataset statistics
python train.py --show-stats
```

### Hospital Configuration

The federated learning setup simulates two hospitals with different dataset access:

**Hospital 1**:
- JSIEC, APTOS2019, MESSIDOR2, IDRiD, OCTID

**Hospital 2**:
- JSIEC, APTOS2019, PAPILA, Glaucoma_fundus, Retina

Modify `HOSPITAL_DATASETS` in `configs/config.py` to change hospital assignments.

## Advanced Configuration

### Model Architecture
- **Backbone**: Vision Transformer Large (vit_large_patch16_224)
- **Input Size**: 224×224
- **Classification Head**: Multi-layer MLP with dropout and layer normalization
- **Number of Classes**: 39 (unified JSIEC taxonomy)

### Federated Learning Parameters
- **Distillation Temperature**: 5.0
- **Knowledge Distillation Weight (α)**: 0.7
- **Feature Dimension**: 1024
- **Early Stopping Patience**: 5 rounds

### Data Augmentation
**Training**:
- RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
- RandomRotation, ColorJitter, RandomAffine
- GaussianBlur, RandomAdjustSharpness

**Evaluation**:
- Resize to 224×224 and normalize

## Evaluation Metrics

The framework evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1-Score**: Macro-averaged F1-score
- **AUC**: Area under ROC curve
- **Specificity**: Macro-averaged specificity
- **Cohen's Kappa**: Inter-rater agreement

## Output Files

### Checkpoints
- `checkpoints_federated/`: Directory containing all training checkpoints
- `best_federated_model.pth`: Best performing global model
- `aggregated_model_round_X.pth`: Model after each federated round

### Logs
Training progress and evaluation results are printed to console with detailed metrics for each dataset and federated round.


### System Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 16GB+ recommended
- **Storage**: Depends on dataset size (typically 50-100GB)

## Citation

If you use this code in your research, please cite our paper: (Details will be provided later)


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **RETFound**: For providing the benchmark datasets and evaluation protocols
- **ATLASS**: For the self-supervised pretrained model
- **timm**: For the Vision Transformer implementation
- **PyTorch**: For the deep learning framework

## Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Contact the authors through the paper correspondence

