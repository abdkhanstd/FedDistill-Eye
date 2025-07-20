"""
Main training script for FedDistill-Eye: Privacy-Preserving Ophthalmic Disease Classification

This script implements federated learning with knowledge distillation for multi-domain 
ophthalmic disease classification while preserving data privacy across different hospitals.

Paper: "Beyond Parameter Sharing: FedDistill-Eye for Privacy-Preserving Ophthalmic Disease Classification"

This file was refactored and cleaned from a single monolithic script to improve:
- Code organization and readability
- Modularity and maintainability
- Error handling and logging
- Separation of concerns

Key Features:
- Federated learning across multiple hospitals
- Knowledge distillation for privacy preservation
- Multi-domain ophthalmic disease classification
- Vision Transformer (ViT) backbone with custom head
- Early stopping and model checkpointing
- Comprehensive evaluation on multiple datasets

Usage:
    python train.py                    # Full federated training
    python train.py --test-only       # Evaluation only
    python train.py --dataset JSIEC   # Test on specific dataset
"""

import torch
import os
import argparse
import copy
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False", category=FutureWarning)

# Import our modules
from configs.config import (
    DEVICE, EPOCHS, PATIENCE, CHECKPOINT_PATH, PRETRAINED_PATH, NUM_CLASSES
)
from src.data.dataset import create_hospital_datasets
from src.models.model import create_base_vit_model, load_pretrained_weights
from src.training.federated import federated_training_round
from src.evaluation.metrics import evaluate_all_datasets, test_on_dataset, print_dataset_statistics


def setup_environment():
    """Setup the training environment and print system information."""
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    # Print system information
    print("="*60)
    print("FedDistill-Eye: Privacy-Preserving Ophthalmic Disease Classification")
    print("="*60)
    print(f"Using device: {DEVICE}")
    
    if torch.cuda.is_available():
        print("\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory // (1024**3)
            print(f"  GPU {i}: {gpu_props.name} ({memory_gb}GB)")
    
    print(f"\nModel Configuration:")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Pretrained model: {PRETRAINED_PATH}")
    print(f"  Checkpoint directory: {CHECKPOINT_PATH}")


def main_training():
    """
    Main federated training loop.
    
    Returns:
        dict: Best global model weights
    """
    print("\n" + "="*60)
    print("STARTING FEDERATED LEARNING TRAINING")
    print("="*60)
    
    # Setup datasets
    print("\nSetting up hospital datasets...")
    hospital_train_loaders, hospital_val_loaders, hospital_mappings = create_hospital_datasets()
    
    if not hospital_train_loaders:
        raise ValueError("No hospital datasets found! Please check your dataset paths.")
    
    # Print dataset sizes
    dataset_sizes = {hospital: len(loader.dataset) for hospital, loader in hospital_train_loaders.items()}
    print(f"\nHospital dataset sizes: {dataset_sizes}")
    
    # Initialize global model
    print(f"\nInitializing global model...")
    global_model = create_base_vit_model(num_classes=NUM_CLASSES)
    
    # Load pretrained weights if available
    if os.path.exists(PRETRAINED_PATH):
        global_model = load_pretrained_weights(global_model, PRETRAINED_PATH)
    else:
        print(f"Warning: Pretrained weights not found at {PRETRAINED_PATH}")
        print("Continuing with random initialization...")
    
    global_model.to(DEVICE)
    
    # Training tracking variables
    best_val_accs = {hospital: 0.0 for hospital in hospital_train_loaders.keys()}
    patience_counters = {hospital: 0 for hospital in hospital_train_loaders.keys()}
    best_global_acc = 0.0
    best_global_weights = None
    
    print(f"\nStarting federated training for {EPOCHS} rounds...")
    
    # Main federated training loop
    for round_num in range(EPOCHS):
        print(f"\n" + "="*60)
        print(f"FEDERATED LEARNING ROUND {round_num + 1}/{EPOCHS}")
        print("="*60)
        
        try:
            # Execute one federated training round
            updated_weights, validation_results = federated_training_round(
                global_model,
                hospital_train_loaders,
                hospital_mappings,
                hospital_val_loaders
            )
            
            # Update global model with new weights
            global_model.load_state_dict(updated_weights)
            
            # Save round checkpoint
            round_checkpoint_path = os.path.join(
                CHECKPOINT_PATH, f'aggregated_model_round_{round_num + 1}.pth'
            )
            torch.save(updated_weights, round_checkpoint_path)
            print(f"Saved round checkpoint: {round_checkpoint_path}")
            
            # Check for early stopping
            early_stop_hospitals = []
            for hospital, val_results in validation_results.items():
                val_acc = val_results.get('accuracy', 0.0) * 100
                
                if val_acc > best_val_accs[hospital]:
                    best_val_accs[hospital] = val_acc
                    patience_counters[hospital] = 0
                    print(f"{hospital} improved! New best: {val_acc:.2f}%")
                else:
                    patience_counters[hospital] += 1
                    print(f"{hospital} patience: {patience_counters[hospital]}/{PATIENCE}")
                
                if patience_counters[hospital] >= PATIENCE:
                    early_stop_hospitals.append(hospital)
            
            # Test aggregated model on all datasets
            print(f"\n--- Testing Aggregated Model (Round {round_num + 1}) ---")
            round_results = evaluate_all_datasets(round_checkpoint_path)
            
            if round_results:
                avg_acc = np.mean([metrics['accuracy'] for metrics in round_results.values()]) * 100
                print(f"\nAverage accuracy across all datasets: {avg_acc:.2f}%")
                
                # Update best global model if improved
                if avg_acc > best_global_acc:
                    best_global_acc = avg_acc
                    best_global_weights = copy.deepcopy(updated_weights)
                    best_model_path = os.path.join(CHECKPOINT_PATH, 'best_federated_model.pth')
                    torch.save(best_global_weights, best_model_path)
                    print(f"NEW BEST MODEL! Saved to: {best_model_path}")
                    print(f"Best average accuracy: {best_global_acc:.2f}%")
            
            # Check for early stopping across all hospitals
            if len(early_stop_hospitals) == len(hospital_train_loaders):
                print(f"\nEarly stopping triggered for all hospitals after round {round_num + 1}")
                break
                
        except Exception as e:
            print(f"Error in round {round_num + 1}: {str(e)}")
            print("Continuing to next round...")
            continue
    
    # Print final results
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    print("\nFinal Best Local Validation Accuracies:")
    for hospital, acc in best_val_accs.items():
        print(f"  {hospital}: {acc:.2f}%")
    
    if best_global_weights:
        print(f"\nBest Global Model Performance: {best_global_acc:.2f}%")
        final_evaluation_path = os.path.join(CHECKPOINT_PATH, 'best_federated_model.pth')
    else:
        print("\nUsing final round model for evaluation...")
        best_global_weights = global_model.state_dict()
        final_evaluation_path = os.path.join(CHECKPOINT_PATH, f'aggregated_model_round_{round_num + 1}.pth')
    
    # Final comprehensive evaluation
    print(f"\n" + "="*60)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("="*60)
    evaluate_all_datasets(final_evaluation_path)
    
    return best_global_weights


def main():
    """Main function handling command line arguments and execution flow."""
    parser = argparse.ArgumentParser(
        description='FedDistill-Eye: Privacy-Preserving Ophthalmic Disease Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                           # Run full federated training
  python train.py --test-only              # Evaluate existing model
  python train.py --test-only --dataset JSIEC  # Test on specific dataset
  python train.py --model-path custom.pth     # Use custom model path
        """
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true', 
        help='Run in test-only mode (skip training)'
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='best_federated_model.pth',
        help='Path to model for testing (default: best_federated_model.pth)'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        help='Specific dataset to test on (if not provided, tests on all datasets)'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show dataset statistics'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Show dataset statistics if requested
    if args.show_stats:
        print_dataset_statistics()
    
    try:
        if args.test_only:
            print(f"\n" + "="*60)
            print("RUNNING IN TEST-ONLY MODE")
            print("="*60)
            
            # Check if model file exists
            if not os.path.exists(args.model_path):
                print(f"Error: Model file '{args.model_path}' not found!")
                print("\nAvailable model files in current directory:")
                model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
                if model_files:
                    for f in model_files:
                        print(f"  {f}")
                else:
                    print("  No .pth files found")
                return 1
            
            # Run evaluation
            if args.dataset:
                print(f"Testing on dataset: {args.dataset}")
                test_on_dataset(args.model_path, args.dataset, verbose=True)
            else:
                print("Testing on all available datasets...")
                evaluate_all_datasets(args.model_path)
        
        else:
            # Run full training pipeline
            try:
                model_weights = main_training()
                print(f"\nTraining completed successfully!")
                return 0
            except KeyboardInterrupt:
                print(f"\nTraining interrupted by user")
                return 1
            except Exception as e:
                print(f"\nTraining failed with error: {str(e)}")
                raise e
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
