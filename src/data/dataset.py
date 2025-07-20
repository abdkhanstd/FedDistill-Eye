"""
Data handling utilities for FedDistill-Eye

This module handles dataset loading, preprocessing, and class mapping for the federated learning setup.
Extracted and cleaned from the original monolithic code for better organization and maintainability.

Key Features:
- Custom dataset class for multi-domain ophthalmic images
- Data transformations for training and evaluation
- Class mapping between different dataset taxonomies
- Hospital-specific dataset organization
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np

from configs.config import (
    DATASET_CLASS_MAPPINGS, VALID_IMAGE_EXTENSIONS, BATCH_SIZE,
    NUM_WORKERS, PREFETCH_FACTOR, PIN_MEMORY, HOSPITAL_DATASETS
)


class FineTuneDataset(Dataset):
    """
    Custom dataset class for handling multiple ophthalmic disease datasets.
    
    This dataset maps different class taxonomies to a unified JSIEC-based classification system.
    It handles various image formats and provides consistent labeling across different datasets.
    """
    
    def __init__(self, dataset_path, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
            split (str): Data split ('train', 'val', 'test')
            transform: Image transformations to apply
        """
        self.root_dir = os.path.join(dataset_path, split)
        self.transform = transform
        self.dataset_name = os.path.basename(dataset_path)
        
        # Use JSIEC as the reference taxonomy
        jsiec_path = os.path.join('datasets', 'JSIEC', 'train')
        if os.path.exists(jsiec_path):
            self.classes = sorted(os.listdir(jsiec_path))
        else:
            # Fallback class list if JSIEC not available
            self.classes = [
                '0.0.Normal', '0.3.DR1', '1.0.DR2', '1.1.DR3', '29.1.Blur fundus with suspected PDR',
                '10.0.Possible glaucoma', '10.1.Optic atrophy', '5.0.CSCR', '8.MH', '6.Maculopathy',
                '29.0.Blur fundus without PDR'
            ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        # Load samples with class mapping
        self._load_samples()
    
    def _load_samples(self):
        """Load image samples and apply class mapping."""
        if self.dataset_name not in DATASET_CLASS_MAPPINGS:
            print(f"Warning: Dataset {self.dataset_name} not in mapping, skipping")
            return
        
        mapping = DATASET_CLASS_MAPPINGS[self.dataset_name]
        
        for class_name in os.listdir(self.root_dir):
            if class_name in mapping:
                mapped_class = mapping[class_name]
                class_dir = os.path.join(self.root_dir, class_name)
                
                if not os.path.isdir(class_dir):
                    continue
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        img_path = os.path.join(class_dir, img_name)
                        if os.path.isfile(img_path):
                            label_idx = self.class_to_idx.get(mapped_class)
                            if label_idx is not None:
                                self.samples.append((img_path, label_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            return image, label


def get_transforms():
    """
    Get data transformations for training and evaluation.
    
    Returns:
        tuple: (train_transform, eval_transform)
    """
    train_transform = transforms.Compose([
        # Data augmentation for training
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomAdjustSharpness(2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        # Simple preprocessing for evaluation
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform


def get_class_mapping_indices(dataset_name):
    """
    Get mapping from dataset-specific class indices to JSIEC indices.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Mapping from local class index to JSIEC class index
    """
    dataset_path = os.path.join('datasets', dataset_name, 'train')
    if not os.path.exists(dataset_path):
        return {}
    
    dataset = FineTuneDataset(
        dataset_path=os.path.join('datasets', dataset_name),
        split='train'
    )
    
    dataset_classes = sorted(os.listdir(dataset_path))
    dataset_to_jsiec_idx = {}
    mapping = DATASET_CLASS_MAPPINGS.get(dataset_name, {})
    
    for idx, orig_class in enumerate(dataset_classes):
        if orig_class in mapping:
            jsiec_class = mapping[orig_class]
            jsiec_idx = dataset.class_to_idx.get(jsiec_class)
            if jsiec_idx is not None:
                dataset_to_jsiec_idx[idx] = jsiec_idx
    
    return dataset_to_jsiec_idx


def get_inverse_mapping(dataset_name):
    """
    Get mapping from JSIEC class indices to dataset-specific indices.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Mapping from JSIEC class index to local class index
    """
    dataset_path = os.path.join('datasets', dataset_name, 'train')
    if not os.path.exists(dataset_path):
        return {}
    
    dataset = FineTuneDataset(
        dataset_path=os.path.join('datasets', dataset_name),
        split='train'
    )
    
    dataset_classes = sorted(os.listdir(dataset_path))
    jsiec_to_dataset = {}
    mapping = DATASET_CLASS_MAPPINGS.get(dataset_name, {})
    
    for orig_class in dataset_classes:
        if orig_class in mapping:
            jsiec_class = mapping[orig_class]
            jsiec_idx = dataset.class_to_idx.get(jsiec_class)
            local_idx = dataset_classes.index(orig_class)
            if jsiec_idx is not None:
                jsiec_to_dataset[jsiec_idx] = local_idx
    
    return jsiec_to_dataset


def create_hospital_datasets():
    """
    Create data loaders for each hospital based on their available datasets.
    
    Returns:
        tuple: (hospital_train_loaders, hospital_val_loaders, hospital_mappings)
    """
    train_transform, eval_transform = get_transforms()
    
    hospital_train_loaders = {}
    hospital_val_loaders = {}
    hospital_mappings = {}

    for hospital, dataset_paths in HOSPITAL_DATASETS.items():
        print(f"Setting up datasets for {hospital}...")
        
        hospital_train_sets = []
        hospital_val_sets = []
        hospital_mapping = {}
        
        for path in dataset_paths:
            dataset_name = os.path.basename(path)
            try:
                # Get class mapping for this dataset
                class_mapping = get_class_mapping_indices(dataset_name)
                hospital_mapping.update(class_mapping)
                
                # Create train and validation datasets
                train_dataset = FineTuneDataset(path, 'train', train_transform)
                val_dataset = FineTuneDataset(path, 'val', eval_transform)
                
                if len(train_dataset.samples) > 0:
                    hospital_train_sets.append(train_dataset)
                    print(f"  Added {dataset_name} training: {len(train_dataset.samples)} samples")
                
                if len(val_dataset.samples) > 0:
                    hospital_val_sets.append(val_dataset)
                    print(f"  Added {dataset_name} validation: {len(val_dataset.samples)} samples")
                    
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue
        
        # Create data loaders if we have datasets
        if hospital_train_sets:
            hospital_train_loaders[hospital] = DataLoader(
                ConcatDataset(hospital_train_sets),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                pin_memory=PIN_MEMORY
            )
            
            if hospital_val_sets:
                hospital_val_loaders[hospital] = DataLoader(
                    ConcatDataset(hospital_val_sets),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    prefetch_factor=PREFETCH_FACTOR,
                    pin_memory=PIN_MEMORY
                )
            
            hospital_mappings[hospital] = hospital_mapping
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    for hospital in hospital_train_loaders:
        train_size = len(hospital_train_loaders[hospital].dataset)
        val_size = len(hospital_val_loaders[hospital].dataset) if hospital in hospital_val_loaders else 0
        print(f"{hospital}:")
        print(f"  Training samples: {train_size}")
        print(f"  Validation samples: {val_size}")
    
    return hospital_train_loaders, hospital_val_loaders, hospital_mappings


class FeatureDataset(Dataset):
    """
    Dataset for storing extracted features and soft labels for distillation.
    """
    
    def __init__(self, features, soft_labels):
        """
        Initialize with pre-extracted features and soft labels.
        
        Args:
            features (torch.Tensor): Extracted features
            soft_labels (torch.Tensor): Soft prediction labels
        """
        self.features = features
        self.soft_labels = soft_labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.soft_labels[idx]
