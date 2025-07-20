"""
Configuration file for FedDistill-Eye: Privacy-Preserving Ophthalmic Disease Classification

This module contains all configuration parameters used throughout the federated learning pipeline.
Organized and cleaned from the original monolithic code for better maintainability.
"""

import torch

# Device Configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
EPOCHS = 50
LOCAL_EPOCHS = 5
PATIENCE = 5
WEIGHT_DECAY = 0.01

# Model Configuration
FEATURE_DIM = 1024
DISTILLATION_TEMP = 5.0
ALPHA = 0.7  # Distillation loss weight

# Paths
CHECKPOINT_PATH = './checkpoints_federated'
PRETRAINED_PATH = 'best_model.pth'

# Hospital Dataset Configuration
# Each hospital has access to different combinations of datasets
HOSPITAL_DATASETS = {
    'Hospital_1': [
        'datasets/JSIEC',
        'datasets/APTOS2019',
        'datasets/MESSIDOR2',
        'datasets/IDRiD',
        'datasets/OCTID'
    ],
    'Hospital_2': [
        'datasets/JSIEC',
        'datasets/APTOS2019',
        'datasets/PAPILA',
        'datasets/Glaucoma_fundus',
        'datasets/Retina'
    ]
}

# Dataset Class Mappings
# Maps each dataset's class names to the unified JSIEC class taxonomy
DATASET_CLASS_MAPPINGS = {
    'JSIEC': {cls: cls for cls in [
        '0.0.Normal', '0.3.DR1', '1.0.DR2', '1.1.DR3', '29.1.Blur fundus with suspected PDR',
        '10.0.Possible glaucoma', '10.1.Optic atrophy', '5.0.CSCR', '8.MH', '6.Maculopathy',
        '29.0.Blur fundus without PDR'
    ]},
    'APTOS2019': {
        'anodr': '0.0.Normal',
        'bmilddr': '0.3.DR1',
        'cmoderatedr': '1.0.DR2',
        'dseveredr': '1.1.DR3',
        'eproliferativedr': '29.1.Blur fundus with suspected PDR'
    },
    'MESSIDOR2': {
        'anodr': '0.0.Normal',
        'bmilddr': '0.3.DR1',
        'cmoderatedr': '1.0.DR2',
        'dseveredr': '1.1.DR3',
        'eproliferativedr': '29.1.Blur fundus with suspected PDR'
    },
    'IDRiD': {
        'anoDR': '0.0.Normal',
        'bmildDR': '0.3.DR1',
        'cmoderateDR': '1.0.DR2',
        'dsevereDR': '1.1.DR3',
        'eproDR': '29.1.Blur fundus with suspected PDR'
    },
    'PAPILA': {
        'anormal': '0.0.Normal',
        'bsuspectglaucoma': '10.0.Possible glaucoma',
        'cglaucoma': '10.1.Optic atrophy'
    },
    'Glaucoma_fundus': {
        'anormal_control': '0.0.Normal',
        'bearly_glaucoma': '10.0.Possible glaucoma',
        'cadvanced_glaucoma': '10.1.Optic atrophy'
    },
    'OCTID': {
        'ANormal': '0.0.Normal',
        'CSR': '5.0.CSCR',
        'Diabetic_retinopathy': '1.0.DR2',
        'Macular_Hole': '8.MH',
        'ARMD': '6.Maculopathy'
    },
    'Retina': {
        'anormal': '0.0.Normal',
        'cglaucoma': '10.1.Optic atrophy',
        'bcataract': '29.0.Blur fundus without PDR',
        'ddretina_disease': '6.Maculopathy'
    }
}

# Data Processing Configuration
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True

# Total number of classes in the unified taxonomy
NUM_CLASSES = 39
