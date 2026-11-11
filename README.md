## Code for our paper:
## "Beyond parameter sharing: a feature distillation framework for privacy-preserving ophthalmic disease classification"

This repository contains the inference code and the trained weights of **FedDistill-Eye**, a federated learning framework with feature distillation for privacy-preserving ophthalmic disease classification across multiple hospitals. With the files here you can reproduce the test results of the paper and classify your own retinal images.

The complete training implementation will be released upon acceptance of the paper.

## Key Features
- **Privacy-Preserving**: No raw images and no model parameters are uploaded by a hospital
- **Feature Distillation**: Hospitals upload high-level features and temperature-scaled soft labels, and the server distils one global classification head from them
- **Multi-Domain**: One model for eight ophthalmic datasets with a unified label space of 39 classes
- **Vision Transformer**: Frozen self-supervised ViT-Large backbone with a hierarchical classification head
- **Reproducible**: One command evaluates the released model and compares the result with the values of the paper

## Method in Brief

1. Every hospital receives the same pretrained ViT-Large backbone. The backbone stays frozen at every site.
2. A hospital trains only the classification head on its own images.
3. For each training image the hospital uploads the 1024-dimensional backbone feature and the soft label of its local head (temperature 5).
4. The server distils a global head from the pooled features and soft labels (KL divergence and cross-entropy on pseudo-labels, with a weight that rises from 0.5 to 0.7).
5. The global head is sent back to the hospitals for the next round.

The released model is the frozen backbone together with the global head. Five global heads are provided, one for each training seed. The paper reports the mean over the five.

## Project Structure

```
FedDistill-Eye/
├── inference.py                  # Evaluation on the test sets and classification of single images
├── feddistill_eye/
│   ├── model.py                  # Backbone, classification head, loading of weights
│   ├── data.py                   # Datasets, class harmonisation, preprocessing
│   ├── metrics.py                # Evaluation metrics of the paper
│   └── class_maps.json           # The 39 global classes and the classes of each dataset
├── weights/
│   └── global_head_seed{0..4}.pth  # The five trained global heads (17 MB each)
├── expected_results.json         # Test results of every head, for comparison
├── requirements.txt              # Python dependencies
└── README.md                     # This file
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
- **JSIEC** (Multi-disease, the reference label space with 39 classes)
- **APTOS2019** (Diabetic Retinopathy)
- **MESSIDOR2** (Diabetic Retinopathy, called MESSIDOR in the paper)
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
│   └── test/
│       ├── 0.0.Normal/
│       ├── 0.3.DR1/
│       ├── 1.0.DR2/
│       └── ...
├── APTOS2019/
│   └── test/
│       ├── anodr/
│       ├── bmilddr/
│       └── ...
└── (other datasets with similar structure)
```

**Important**: Keep the folder names of the benchmark. Inference needs only the `test/` split of each dataset.

### 3. Weights

#### Frozen Backbone
Download the self-supervised pretrained model (`best_model.pth`) from:
- **Repository**: [ATLASS](https://github.com/abdkhanstd/ATLASS)

Place the file at `weights/best_model.pth`. Its SHA-256 checksum is
`e090e85cfaa6d3d11fc4b14603154fa6d4b4a70581e6323f9d5b4eb2e2bfbb19`.

#### Global Heads
The five global heads of the paper are part of this repository (`weights/global_head_seed0.pth` to `weights/global_head_seed4.pth`). Nothing else has to be downloaded.

## Usage

### Reproduce the Results of the Paper

```bash
# Evaluate the five global heads on the test split of the eight datasets
python inference.py --data-root datasets --backbone weights/best_model.pth --check
```

The script prints one table for each head, the mean over the five heads, and the largest difference to `expected_results.json`. On one GPU the run takes about three minutes.

Expected output (mean over the five heads, test split, values in %):

| Dataset | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| APTOS2019 | 86.75 | 77.00 | 73.38 | 74.77 | 94.92 |
| Glaucoma Fundus | 84.82 | 81.33 | 82.79 | 81.79 | 94.93 |
| IDRiD | 78.25 | 76.05 | 70.01 | 71.84 | 90.93 |
| JSIEC | 87.66 | 89.57 | 79.74 | 81.93 | 98.83 |
| MESSIDOR | 74.68 | 74.78 | 61.94 | 65.87 | 88.96 |
| OCTID | 99.43 | 99.68 | 98.82 | 99.23 | 99.80 |
| PAPILA | 83.27 | 77.88 | 68.66 | 71.48 | 90.86 |
| Retina | 73.48 | 73.30 | 69.22 | 70.22 | 90.07 |
| **Average** | **83.54** | **81.20** | **75.57** | **77.14** | **93.66** |

The average accuracy is 83.54 ± 0.28 % (mean and standard deviation over the five heads).

### Test on Specific Datasets or Heads

```bash
# Only some datasets
python inference.py --data-root datasets --backbone weights/best_model.pth --datasets JSIEC OCTID

# Only one head, with the metrics written to a file
python inference.py --data-root datasets --backbone weights/best_model.pth \
    --heads weights/global_head_seed0.pth --out results_seed0.json
```

### Classify Your Own Images

```bash
# Prediction over all 39 classes
python inference.py --backbone weights/best_model.pth --images eye1.jpg eye2.png

# Prediction over the classes of one dataset, for example the five DR grades of APTOS2019
python inference.py --backbone weights/best_model.pth --images eye1.jpg --dataset APTOS2019
```

### Single-File Model

```bash
# Write the backbone and one head into one checkpoint file
python inference.py --backbone weights/best_model.pth --heads weights/global_head_seed0.pth \
    --save-full-model FedDistill-Eye_full_model_seed0.pth

# Use it instead of --backbone and --heads
python inference.py --data-root datasets --model FedDistill-Eye_full_model_seed0.pth
```

## Evaluation Protocol

- **Class restriction**: For an image of a dataset, the prediction is taken over the classes of that dataset. Every class of a dataset is mapped to one of the 39 global classes (`feddistill_eye/class_maps.json`).
- **Macro averages**: Precision, recall, F1-score, specificity and AUC are macro averages over the classes of a dataset.
- **Average over datasets**: The reported average is the unweighted mean over the eight datasets.
- **Seeds**: Every value in the paper is the mean over the five global heads.

## Model

- **Backbone**: Vision Transformer Large (vit_large_patch16_224), self-supervised pretrained, frozen
- **Input Size**: 224×224
- **Uploaded Feature**: 1024-dimensional class token of the backbone
- **Classification Head**: Multi-layer MLP with dropout and layer normalization (1024, 2048, 1024, 39)
- **Number of Classes**: 39 (unified JSIEC taxonomy)
- **Preprocessing**: Resize to 224×224 and normalize with the ImageNet mean and standard deviation

## Training Setup of the Released Heads

The federated setup simulates two hospitals with different dataset access:

**Hospital 1**:
- JSIEC, APTOS2019, MESSIDOR2, IDRiD, OCTID

**Hospital 2**:
- JSIEC, APTOS2019, PAPILA, Glaucoma_fundus, Retina

- **Local Training**: 5 epochs for each round, batch size 64, learning rate 5e-5
- **Distillation Temperature**: 5.0
- **Distillation Weight**: Rises from 0.5 to 0.7 during the distillation at the server
- **Feature Dimension**: 1024
- **Rounds**: At most 50, with early stopping (patience 5 rounds)

## Evaluation Metrics

The script reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1-Score**: Macro-averaged F1-score
- **AUC**: Macro-averaged one-vs-rest area under the ROC curve
- **Specificity**: Macro-averaged specificity
- **Cohen's Kappa**: Agreement between prediction and label beyond chance

### System Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher (tested with 2.5)
- **timm**: 0.9 or higher (tested with 1.0.13)
- **GPU**: Recommended (CUDA-compatible). The script also runs on a CPU
- **Storage**: 1.3 GB for the weights, plus the datasets

## Citation

If you use this code in your research, please cite our paper: (Details will be provided later)

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
