"""
Utility script for dataset validation and statistics
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import print_dataset_statistics
from configs.config import HOSPITAL_DATASETS, DATASET_CLASS_MAPPINGS


def validate_dataset_structure():
    """Validate that datasets are properly structured."""
    print("Validating dataset structure...")
    
    issues = []
    
    # Check if datasets directory exists
    if not os.path.exists('datasets'):
        issues.append("❌ 'datasets' directory not found")
        return issues
    
    # Check each required dataset
    all_datasets = set()
    for hospital_datasets in HOSPITAL_DATASETS.values():
        for dataset_path in hospital_datasets:
            dataset_name = os.path.basename(dataset_path)
            all_datasets.add(dataset_name)
    
    for dataset_name in all_datasets:
        dataset_path = os.path.join('datasets', dataset_name)
        
        if not os.path.exists(dataset_path):
            issues.append(f"❌ Dataset '{dataset_name}' not found at {dataset_path}")
            continue
        
        # Check splits
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                issues.append(f"❌ Split '{split}' not found for dataset '{dataset_name}'")
                continue
            
            # Check if split has class folders
            class_folders = [d for d in os.listdir(split_path) 
                           if os.path.isdir(os.path.join(split_path, d))]
            
            if not class_folders:
                issues.append(f"❌ No class folders found in {split_path}")
            else:
                # Check if classes have images
                empty_classes = []
                for class_name in class_folders:
                    class_path = os.path.join(split_path, class_name)
                    images = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                    if not images:
                        empty_classes.append(class_name)
                
                if empty_classes:
                    issues.append(f"❌ Empty class folders in {split_path}: {empty_classes}")
        
        # Check class mapping
        if dataset_name not in DATASET_CLASS_MAPPINGS:
            issues.append(f"❌ No class mapping found for dataset '{dataset_name}'")
    
    if not issues:
        print("✅ All dataset validation checks passed!")
    
    return issues


def check_pretrained_models():
    """Check if pretrained models are available."""
    print("\nChecking pretrained models...")
    
    issues = []
    
    # Check self-supervised pretrained model
    if not os.path.exists('best_model.pth'):
        issues.append("❌ 'best_model.pth' not found (download from ATLASS repository)")
    else:
        print("✅ Self-supervised pretrained model found")
    
    # Check for existing trained models
    checkpoint_dir = 'checkpoints_federated'
    if os.path.exists(checkpoint_dir):
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if model_files:
            print(f"✅ Found {len(model_files)} checkpoint files")
        else:
            print("ℹ️  No checkpoint files found (normal for first run)")
    
    return issues


def main():
    """Main validation function."""
    print("="*50)
    print("FedDistill-Eye Dataset Validation")
    print("="*50)
    
    all_issues = []
    
    # Validate dataset structure
    dataset_issues = validate_dataset_structure()
    all_issues.extend(dataset_issues)
    
    # Check pretrained models
    model_issues = check_pretrained_models()
    all_issues.extend(model_issues)
    
    # Print dataset statistics if datasets exist
    if not dataset_issues:
        print_dataset_statistics()
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    if all_issues:
        print("❌ Issues found:")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nPlease resolve these issues before training.")
        return 1
    else:
        print("✅ All validation checks passed!")
        print("You're ready to start training!")
        return 0


if __name__ == "__main__":
    exit(main())
