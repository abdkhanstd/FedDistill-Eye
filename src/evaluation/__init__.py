"""
__init__.py for evaluation package
"""

from .metrics import evaluate_model, test_on_dataset, evaluate_all_datasets, print_dataset_statistics

__all__ = ['evaluate_model', 'test_on_dataset', 'evaluate_all_datasets', 'print_dataset_statistics']
