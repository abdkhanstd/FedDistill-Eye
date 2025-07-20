"""
Training utilities for FedDistill-Eye

This module contains the core federated learning training logic including local training,
feature distillation, and global model aggregation.

Extracted and cleaned from the original monolithic code for better maintainability.
Key improvements:
- Separated training logic from evaluation
- Better error handling and logging
- Modular design for easy extension
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

from configs.config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, LOCAL_EPOCHS, 
    DISTILLATION_TEMP, ALPHA, BATCH_SIZE
)
from src.models.model import extract_features_and_predictions
from src.data.dataset import FeatureDataset


def train_local_model(model, dataloader, hospital_mapping, epochs=LOCAL_EPOCHS):
    """
    Train a local model on hospital-specific data.
    
    This function performs local training for a specific hospital using their available datasets.
    It uses different learning rates for the backbone and classification head.
    
    Args:
        model (torch.nn.Module): Local model to train
        dataloader (torch.utils.data.DataLoader): Training data loader
        hospital_mapping (dict): Class mapping for this hospital
        epochs (int): Number of local training epochs
        
    Returns:
        tuple: (extracted_features, soft_predictions) for distillation
    """
    model.train()
    
    # Set up optimizer with different learning rates for backbone and head
    optimizer = optim.AdamW([
        {
            'params': [p for n, p in model.named_parameters() if 'head' not in n], 
            'lr': LEARNING_RATE * 0.1,  # Lower LR for pretrained backbone
            'weight_decay': WEIGHT_DECAY
        },
        {
            'params': model.head.parameters(), 
            'lr': LEARNING_RATE,  # Higher LR for new head
            'weight_decay': WEIGHT_DECAY * 0.1
        }
    ])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting local training for {epochs} epochs...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        
        # Create progress bar for this epoch
        pbar = tqdm(dataloader, desc=f"Local Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use mixed precision training for efficiency
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / num_batches
        print(f"Local Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    print("Local training completed. Extracting features for distillation...")
    
    # Extract features and soft predictions for knowledge distillation
    features, soft_labels = extract_features_and_predictions(
        model, dataloader, DISTILLATION_TEMP
    )
    
    print(f"Extracted {features.shape[0]} feature samples for distillation")
    
    return features, soft_labels


def distill_global_model(global_model, local_features_list, local_soft_labels_list, val_loader=None):
    """
    Perform knowledge distillation to update the global model.
    
    This function aggregates knowledge from multiple hospitals by distilling
    their learned features and predictions into a single global model.
    
    Args:
        global_model (torch.nn.Module): Global model to update
        local_features_list (list): List of feature tensors from local models
        local_soft_labels_list (list): List of soft label tensors from local models
        val_loader (DataLoader, optional): Validation data loader for fine-tuning
        
    Returns:
        dict: Updated global model state dictionary
    """
    print("Starting global model distillation...")
    
    global_model.train()
    
    # Optimizer for global model (only update the head during distillation)
    optimizer = optim.AdamW(
        global_model.head.parameters(), 
        lr=LEARNING_RATE * 2,  # Higher learning rate for distillation
        weight_decay=WEIGHT_DECAY
    )
    
    # Loss functions for distillation
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Combine all local features and soft labels
    all_features = torch.cat(local_features_list)
    all_soft_labels = torch.cat(local_soft_labels_list)
    
    print(f"Combined {all_features.shape[0]} samples from {len(local_features_list)} hospitals")
    
    # Create distillation dataset and loader
    distill_dataset = FeatureDataset(all_features, all_soft_labels)
    distill_loader = DataLoader(
        distill_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Distillation training
    distill_epochs = LOCAL_EPOCHS * 2
    print(f"Running distillation for {distill_epochs} epochs...")
    
    for epoch in range(distill_epochs):
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(distill_loader, desc=f"Distillation Epoch {epoch+1}/{distill_epochs}")
        
        for features, soft_labels in pbar:
            features = features.to(DEVICE)
            soft_labels = soft_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward pass through global model head
                outputs = global_model.head(features)
                
                # Compute distillation loss (KL divergence)
                global_probs = torch.log_softmax(outputs / DISTILLATION_TEMP, dim=1)
                kl_loss = kl_criterion(global_probs, soft_labels)
                
                # Compute classification loss using pseudo labels
                pseudo_labels = torch.argmax(soft_labels, dim=1)
                ce_loss = ce_criterion(outputs, pseudo_labels)
                
                # Combined loss with weighting
                total_loss = ALPHA * kl_loss + (1 - ALPHA) * ce_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Total Loss': f'{total_loss.item():.4f}',
                'KL Loss': f'{kl_loss.item():.4f}',
                'CE Loss': f'{ce_loss.item():.4f}'
            })
        
        avg_loss = running_loss / num_batches
        print(f"Distillation Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Optional fine-tuning on validation data
    if val_loader is not None:
        print("Fine-tuning global model on validation data...")
        _fine_tune_on_validation(global_model, val_loader, optimizer, ce_criterion)
    
    print("Global model distillation completed!")
    
    return global_model.state_dict()


def _fine_tune_on_validation(model, val_loader, optimizer, criterion, epochs=3):
    """
    Fine-tune the global model on validation data.
    
    Args:
        model (torch.nn.Module): Model to fine-tune
        val_loader (DataLoader): Validation data loader
        optimizer: Optimizer instance
        criterion: Loss criterion
        epochs (int): Number of fine-tuning epochs
    """
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Validation Fine-tune {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / num_batches
        print(f"Validation Fine-tune Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")


def federated_training_round(global_model, hospital_loaders, hospital_mappings, val_loaders=None):
    """
    Execute one round of federated training.
    
    Args:
        global_model (torch.nn.Module): Current global model
        hospital_loaders (dict): Training data loaders for each hospital
        hospital_mappings (dict): Class mappings for each hospital
        val_loaders (dict, optional): Validation data loaders
        
    Returns:
        tuple: (updated_global_weights, local_validation_results)
    """
    print(f"\n{'='*50}")
    print("Starting federated training round")
    print(f"{'='*50}")
    
    local_features_list = []
    local_soft_labels_list = []
    validation_results = {}
    
    # Get current global weights
    global_weights = global_model.state_dict()
    
    # Train each hospital's local model
    for hospital, loader in hospital_loaders.items():
        print(f"\n--- Training {hospital} ---")
        print(f"Dataset size: {len(loader.dataset)} samples")
        
        # Create local model and load global weights
        from src.models.model import create_base_vit_model
        from configs.config import NUM_CLASSES
        
        client_model = create_base_vit_model(num_classes=NUM_CLASSES)
        client_model.load_state_dict(global_weights)
        client_model.to(DEVICE)
        
        # Local training
        features, soft_labels = train_local_model(
            client_model, 
            loader,
            hospital_mappings[hospital]
        )
        
        # Evaluate on local validation data if available
        if val_loaders and hospital in val_loaders:
            from src.evaluation.metrics import evaluate_model
            val_metrics = evaluate_model(client_model, val_loaders[hospital])
            validation_results[hospital] = val_metrics
            print(f"{hospital} Validation Accuracy: {val_metrics['accuracy']*100:.2f}%")
        
        # Store features and labels for distillation
        local_features_list.append(features)
        local_soft_labels_list.append(soft_labels)
        
        # Clean up memory
        del client_model
        torch.cuda.empty_cache()
    
    # Global model distillation
    print(f"\n--- Global Model Distillation ---")
    val_loader = val_loaders.get('Hospital_1') if val_loaders else None
    updated_weights = distill_global_model(
        global_model, 
        local_features_list, 
        local_soft_labels_list,
        val_loader=val_loader
    )
    
    return updated_weights, validation_results
