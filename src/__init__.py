"""
FedDistill-Eye: Privacy-Preserving Ophthalmic Disease Classification

This package contains the implementation of federated learning with knowledge distillation
for multi-domain ophthalmic disease classification.
"""

__version__ = "1.0.0"
__author__ = "FedDistill-Eye Team"

# Import main components
from . import data, models, training, evaluation

__all__ = ['data', 'models', 'training', 'evaluation']
