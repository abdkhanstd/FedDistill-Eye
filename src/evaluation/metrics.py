"""
Evaluation utilities for FedDistill-Eye

This module contains functions for evaluating model performance using various metrics
and testing on different datasets.

Extracted and cleaned from the original monolithic code for better organization.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, cohen_kappa_score,
    confusion_matrix
)
from torch.utils.data import DataLoader
import os

from configs.config import DEVICE, BATCH_SIZE
from src.data.dataset import FineTuneDataset, get_transforms, get_inverse_mapping
from src.models.model import create_base_vit_model, adapt_model_for_dataset


def evaluate_model(model, test_loader, label_mapping=None):
    """
    Evaluate model performance on a test dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): Test data loader
        label_mapping (dict, optional): Label mapping for class conversion
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Apply label mapping if provided
            if label_mapping:
                inv_mapping = {v: k for k, v in label_mapping.items()}
                mapped_preds = torch.tensor([
                    inv_mapping.get(p.item(), p.item()) for p in preds
                ])
                preds = mapped_preds
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['kappa'] = cohen_kappa_score(all_labels, all_preds)
    except Exception as e:
        print(f"Error calculating basic metrics: {e}")
        # Set default values
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'kappa']:
            metrics[metric] = 0.0
    
    # Calculate AUC (handle potential errors)
    try:
        if all_probs.shape[1] > 2:  # Multi-class
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        else:  # Binary
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError as e:
        print(f"Warning: Could not calculate AUC - {str(e)}")
        metrics['auc'] = 0.0
    
    # Calculate specificity
    try:
        cm = confusion_matrix(all_labels, all_preds)
        specificities = calculate_specificity_from_cm(cm)
        metrics['specificity'] = np.mean(specificities)
    except Exception as e:
        print(f"Warning: Could not calculate specificity - {str(e)}")
        metrics['specificity'] = 0.0
    
    return metrics


def calculate_specificity_from_cm(cm):
    """
    Calculate specificity for each class from confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        
    Returns:
        list: Specificity values for each class
    """
    n_classes = cm.shape[0]
    specificities = []
    
    for i in range(n_classes):
        # True negatives: sum of all entries except row i and column i
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        # False positives: sum of column i excluding diagonal element
        fp = np.sum(np.delete(cm[:, i], i))
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    return specificities


def test_on_dataset(model_path, dataset_name, verbose=True):
    """
    Test a trained model on a specific dataset.
    
    Args:
        model_path (str): Path to the trained model
        dataset_name (str): Name of the dataset to test on
        verbose (bool): Whether to print detailed results
        
    Returns:
        dict: Evaluation metrics
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Testing on {dataset_name}")
        print(f"{'='*50}")
    
    # Get data transforms
    _, eval_transform = get_transforms()
    
    # Create test dataset
    test_dataset_path = os.path.join('datasets', dataset_name)
    if not os.path.exists(test_dataset_path):
        print(f"Error: Dataset path {test_dataset_path} not found")
        return {}
    
    test_dataset = FineTuneDataset(
        test_dataset_path,
        'test',
        eval_transform
    )
    
    if len(test_dataset.samples) == 0:
        print(f"Warning: No test samples found for {dataset_name}")
        return {}
    
    # Get class mappings
    jsiec_to_dataset = get_inverse_mapping(dataset_name)
    if not jsiec_to_dataset:
        print(f"Warning: No class mapping found for {dataset_name}")
        return {}
    
    num_classes = len(set(jsiec_to_dataset.values()))
    
    if verbose:
        print(f"Test samples: {len(test_dataset.samples)}")
        print(f"Number of classes: {num_classes}")
    
    # Create and load model
    test_model = create_base_vit_model(num_classes=num_classes)
    
    try:
        # Load model weights
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        
        # Adapt weights for this dataset
        adapted_state_dict = adapt_model_for_dataset(
            state_dict, dataset_name, jsiec_to_dataset
        )
        
        test_model.load_state_dict(adapted_state_dict)
        test_model.to(DEVICE)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    metrics = evaluate_model(test_model, test_loader, jsiec_to_dataset)
    
    if verbose:
        print(f"\n{dataset_name} Results:")
        print(f"{'Metric':<15} {'Value':<10}")
        print("-" * 25)
        for metric, value in metrics.items():
            print(f"{metric.capitalize():<15} {value*100:>9.2f}%")
    
    return metrics


def evaluate_all_datasets(model_path, datasets_dir='datasets'):
    """
    Evaluate a model on all available datasets.
    
    Args:
        model_path (str): Path to the trained model
        datasets_dir (str): Directory containing all datasets
        
    Returns:
        dict: Results for each dataset
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVALUATION ON ALL DATASETS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return {}
    
    if not os.path.exists(datasets_dir):
        print(f"Error: Datasets directory {datasets_dir} not found")
        return {}
    
    results = {}
    dataset_list = []
    
    # Get list of available datasets
    for dataset in sorted(os.listdir(datasets_dir)):
        dataset_path = os.path.join(datasets_dir, dataset)
        if dataset != 'Combined' and os.path.isdir(dataset_path):
            test_path = os.path.join(dataset_path, 'test')
            if os.path.exists(test_path):
                dataset_list.append(dataset)
    
    print(f"Found {len(dataset_list)} datasets to evaluate: {', '.join(dataset_list)}")
    
    # Evaluate each dataset
    for dataset in dataset_list:
        try:
            metrics = test_on_dataset(model_path, dataset, verbose=True)
            if metrics:
                results[dataset] = metrics
            else:
                print(f"Skipping {dataset} due to errors")
        except Exception as e:
            print(f"Error testing {dataset}: {str(e)}")
            continue
    
    # Print summary results
    if results:
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Header
        print(f"{'Dataset':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC':>10}")
        print("-" * 80)
        
        # Results for each dataset
        for dataset, metrics in results.items():
            print(f"{dataset:<20} "
                  f"{metrics.get('accuracy', 0)*100:>10.2f} "
                  f"{metrics.get('precision', 0)*100:>10.2f} "
                  f"{metrics.get('recall', 0)*100:>10.2f} "
                  f"{metrics.get('f1', 0)*100:>10.2f} "
                  f"{metrics.get('auc', 0)*100:>10.2f}")
        
        print("-" * 80)
        
        # Average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa', 'specificity']:
            values = [results[dataset].get(metric, 0) for dataset in results]
            avg_metrics[metric] = np.mean(values) if values else 0
        
        print("\nAVERAGE METRICS ACROSS ALL DATASETS:")
        for metric, value in avg_metrics.items():
            print(f"{metric.capitalize()}: {value*100:.2f}%")
        
        print(f"\n{'='*80}")
    
    return results


def print_dataset_statistics(datasets_dir='datasets'):
    """
    Print statistics about available datasets.
    
    Args:
        datasets_dir (str): Directory containing datasets
    """
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    
    if not os.path.exists(datasets_dir):
        print(f"Datasets directory {datasets_dir} not found")
        return
    
    for dataset_name in sorted(os.listdir(datasets_dir)):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if os.path.isdir(dataset_path):
            print(f"\n{dataset_name}:")
            
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(dataset_path, split)
                if os.path.exists(split_path):
                    total_samples = 0
                    class_counts = {}
                    
                    for class_name in os.listdir(split_path):
                        class_path = os.path.join(split_path, class_name)
                        if os.path.isdir(class_path):
                            count = len([f for f in os.listdir(class_path) 
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
                            class_counts[class_name] = count
                            total_samples += count
                    
                    print(f"  {split.capitalize()}: {total_samples} samples across {len(class_counts)} classes")
                    if len(class_counts) <= 10:  # Show class distribution if not too many classes
                        for class_name, count in sorted(class_counts.items()):
                            print(f"    {class_name}: {count}")
