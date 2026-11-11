"""Datasets, class harmonisation and preprocessing.

The eight datasets use the folder layout of the RETFound benchmark:

    <data-root>/<dataset>/<split>/<class folder>/<image>

Every class of a dataset is mapped to one of the 39 global classes (the JSIEC label
space). The mapping is stored in class_maps.json next to this file.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Order used in the tables of the paper.
DATASETS = ['APTOS2019', 'Glaucoma_fundus', 'IDRiD', 'JSIEC', 'MESSIDOR2', 'OCTID', 'PAPILA', 'Retina']

# Name of each dataset in the paper.
PAPER_NAME = {'APTOS2019': 'APTOS2019', 'Glaucoma_fundus': 'Glaucoma Fundus', 'IDRiD': 'IDRiD',
              'JSIEC': 'JSIEC', 'MESSIDOR2': 'MESSIDOR', 'OCTID': 'OCTID', 'PAPILA': 'PAPILA',
              'Retina': 'Retina'}

VALID_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_maps.json'), encoding='utf-8') as _f:
    _MAPS = json.load(_f)

GLOBAL_CLASSES = _MAPS['_global_classes']                              # 39 names
LOCAL_CLASSES = {d: _MAPS[d]['local_classes'] for d in DATASETS}       # class folders of a dataset
GLOBAL_INDEX = {d: _MAPS[d]['global_index'] for d in DATASETS}         # their global class index


def list_samples(data_root, dataset, split='test'):
    """(path, local class index) of every image of one dataset split."""
    root = os.path.join(data_root, dataset, split)
    if not os.path.isdir(root):
        raise FileNotFoundError('folder not found: %s' % root)
    samples = []
    for label, cname in enumerate(LOCAL_CLASSES[dataset]):
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            path = os.path.join(cdir, fname)
            if fname.lower().endswith(VALID_EXT) and os.path.isfile(path):
                samples.append((path, label))
    return samples


_RESIZE = transforms.Resize((224, 224))


def load_image(path):
    """Evaluation transform of the paper: resize to 224 x 224, scale to [0, 1], normalise."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(_RESIZE(img), dtype=np.uint8)
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return x.sub_(mean).div_(std)


class ImageList(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return load_image(path), label
