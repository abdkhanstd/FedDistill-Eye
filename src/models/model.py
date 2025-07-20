"""
Model definitions for FedDistill-Eye

This module contains the Vision Transformer model architecture and utilities for loading pretrained weights.
Extracted and cleaned from the original monolithic code for better organization.

Key Features:
- Vision Transformer (ViT) with custom classification head
- Pretrained weight loading utilities
- Feature extraction capabilities for distillation
"""

import torch
import torch.nn as nn
import timm
from configs.config import DEVICE


def create_base_vit_model(num_classes):
    """
    Create a Vision Transformer model with custom classification head.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        torch.nn.Module: ViT model with custom head
    """
    # Create ViT Large model without pretrained weights initially
    model = timm.create_model('vit_large_patch16_224', pretrained=False)
    in_features = model.head.in_features
    
    # Replace the head with a custom multi-layer classifier
    model.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 2048),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.LayerNorm(2048),
        nn.Linear(2048, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.LayerNorm(1024),
        nn.Linear(1024, num_classes)
    )
    
    return model


def load_pretrained_weights(model, pretrained_path):
    """
    Load pretrained weights into the model, excluding the classification head.
    
    This function loads weights from a pretrained model while keeping the backbone
    weights and initializing a new classification head for the target task.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        pretrained_path (str): Path to the pretrained weights file
        
    Returns:
        torch.nn.Module: Model with loaded pretrained weights
    """
    try:
        # Load pretrained weights
        pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
        model_dict = model.state_dict()
        
        # Filter out head weights and only keep backbone weights
        pretrained_dict_filtered = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and 'head' not in k
        }
        
        # Update model dictionary with pretrained weights
        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        
        print(f"Successfully loaded pretrained weights from {pretrained_path}")
        print(f"Loaded {len(pretrained_dict_filtered)} layers from pretrained model")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights from {pretrained_path}: {e}")
        print("Continuing with random initialization...")
    
    return model


def extract_features_and_predictions(model, dataloader, distillation_temp=5.0):
    """
    Extract features and soft predictions from a model.
    
    This function extracts intermediate features from the ViT model (before the final
    classification layer) and computes softmax predictions for knowledge distillation.
    
    Args:
        model (torch.nn.Module): Model to extract features from
        dataloader (torch.utils.data.DataLoader): Data loader
        distillation_temp (float): Temperature for softmax scaling
        
    Returns:
        tuple: (features, soft_labels) as torch tensors
    """
    model.eval()
    features = []
    soft_labels = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)
            
            # Forward pass through ViT backbone
            x = model.patch_embed(inputs)
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + model.pos_embed
            x = model.pos_drop(x)
            
            # Pass through transformer blocks
            for blk in model.blocks:
                x = blk(x)
            x = model.norm(x)
            
            # Extract features from CLS token
            cls_features = x[:, 0]  # CLS token features
            
            # Pass through head (except final layer) to get refined features
            features_out = model.head[:-1](cls_features)
            
            # Get final predictions for soft labels
            logits = model.head[-1](features_out)
            probs = torch.softmax(logits / distillation_temp, dim=1)
            
            features.append(features_out.cpu())
            soft_labels.append(probs.cpu())
    
    return torch.cat(features), torch.cat(soft_labels)


def adapt_model_for_dataset(model_state_dict, dataset_name, jsiec_to_dataset_mapping):
    """
    Adapt a trained model for evaluation on a specific dataset.
    
    This function modifies the final classification layer to match the number of classes
    in the target dataset and maps the learned JSIEC classes to dataset-specific classes.
    
    Args:
        model_state_dict (dict): State dictionary of the trained model
        dataset_name (str): Name of the target dataset
        jsiec_to_dataset_mapping (dict): Mapping from JSIEC to dataset classes
        
    Returns:
        dict: Adapted state dictionary
    """
    num_classes = len(set(jsiec_to_dataset_mapping.values()))
    new_state_dict = {}
    
    for k, v in model_state_dict.items():
        if 'head.10.weight' in k:  # Final classification layer weights
            # Create new weight matrix for the target dataset
            new_weights = torch.zeros((num_classes, v.size(1)), device=v.device)
            for jsiec_idx, dataset_idx in jsiec_to_dataset_mapping.items():
                if jsiec_idx < v.size(0):
                    new_weights[dataset_idx] = v[jsiec_idx]
            new_state_dict[k] = new_weights
            
        elif 'head.10.bias' in k:  # Final classification layer bias
            # Create new bias vector for the target dataset
            new_bias = torch.zeros(num_classes, device=v.device)
            for jsiec_idx, dataset_idx in jsiec_to_dataset_mapping.items():
                if jsiec_idx < v.size(0):
                    new_bias[dataset_idx] = v[jsiec_idx]
            new_state_dict[k] = new_bias
            
        else:
            # Keep all other layers unchanged
            new_state_dict[k] = v
    
    return new_state_dict
