"""Evaluation metrics as used in the paper.

For an image of dataset D the prediction is the arg-max over the classes of D (the
classes that exist in that dataset). Precision, recall, F1, specificity and AUC are
macro averages over the classes. The reported average is the unweighted mean over the
eight datasets.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'npv', 'kappa']


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_auc(y, prob):
    """Macro one-vs-rest AUC over the classes that occur in y."""
    aucs = []
    for k in range(prob.shape[1]):
        pos = (y == k)
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos, prob[:, k]))
    return float(np.mean(aucs)) if aucs else float('nan')


def compute_metrics(y, pred, prob):
    labels = np.unique(np.concatenate([y, pred]))
    cm = confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        npv = np.where(tn + fn > 0, tn / (tn + fn), 0.0)
    out = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, average='macro', zero_division=0),
        'recall': recall_score(y, pred, average='macro', zero_division=0),
        'f1': f1_score(y, pred, average='macro', zero_division=0),
        'auc': macro_auc(y, prob),
        'specificity': float(spec.mean()),
        'npv': float(npv.mean()),
        'kappa': cohen_kappa_score(y, pred),
    }
    return {k: float(v) for k, v in out.items()}


def evaluate_dataset(logits, y, global_index):
    """Metrics of one dataset from the 39-way logits of its images."""
    restricted = logits[:, np.asarray(global_index)]     # only the classes of this dataset compete
    return compute_metrics(y, restricted.argmax(1), softmax(restricted))
