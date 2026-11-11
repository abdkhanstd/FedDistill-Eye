"""FedDistill-Eye: inference with the released model.

1. Evaluate on the test split of the eight datasets (protocol of the paper):

       python inference.py --data-root datasets --backbone weights/best_model.pth

   Each of the five global heads (five training seeds) is evaluated. The paper reports the
   mean over the five heads. Add --check to compare with expected_results.json.

2. Classify single images:

       python inference.py --backbone weights/best_model.pth --images eye1.jpg eye2.png
       python inference.py --backbone weights/best_model.pth --images eye1.jpg --dataset APTOS2019

3. A single-file checkpoint (backbone and one head) can be used instead of --backbone:

       python inference.py --data-root datasets --model FedDistill-Eye_full_model_seed0.pth
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from feddistill_eye.data import (DATASETS, GLOBAL_CLASSES, GLOBAL_INDEX, LOCAL_CLASSES, PAPER_NAME,
                                 ImageList, list_samples, load_image)
from feddistill_eye.metrics import METRICS, evaluate_dataset, softmax
from feddistill_eye.model import backbone_features, build_model, load_backbone, load_full_model, load_head

HERE = os.path.dirname(os.path.abspath(__file__))
SHOWN = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'kappa']


def parse_args():
    ap = argparse.ArgumentParser(description='FedDistill-Eye inference')
    ap.add_argument('--data-root', default='datasets', help='folder that holds the dataset folders')
    ap.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    ap.add_argument('--split', default='test')
    ap.add_argument('--backbone', default=os.path.join(HERE, 'weights', 'best_model.pth'),
                    help='frozen pretrained backbone (best_model.pth)')
    ap.add_argument('--heads', nargs='+', default=None,
                    help='global head files (default: weights/global_head_seed*.pth)')
    ap.add_argument('--model', default=None, help='single-file checkpoint with backbone and head')
    ap.add_argument('--images', nargs='+', default=None, help='classify these image files')
    ap.add_argument('--dataset', default=None, choices=DATASETS,
                    help='with --images: only the classes of this dataset compete')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None, help='write the metrics to this JSON file')
    ap.add_argument('--check', action='store_true', help='compare with expected_results.json')
    ap.add_argument('--save-full-model', default=None,
                    help='write backbone and the first head into one checkpoint file')
    return ap.parse_args()


def load_models(args):
    """Returns the backbone model and a dict {name: head}."""
    device = torch.device(args.device)
    if args.model:
        model = load_full_model(args.model)
        heads = {os.path.basename(args.model): model.head}
    else:
        if not os.path.isfile(args.backbone):
            raise FileNotFoundError('backbone not found: %s (see README, section Weights)' % args.backbone)
        model = load_backbone(build_model(), args.backbone)
        paths = args.heads or sorted(glob.glob(os.path.join(HERE, 'weights', 'global_head_seed*.pth')))
        if not paths:
            raise FileNotFoundError('no global head found in weights/')
        heads = {os.path.basename(p): load_head(p) for p in paths}
    model = model.to(device).eval()
    heads = {k: h.to(device).eval() for k, h in heads.items()}
    return model, heads, device


@torch.inference_mode()
def features_of(model, samples, args, device):
    loader = DataLoader(ImageList(samples), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    feats = [backbone_features(model, x.to(device)).float() for x, _ in loader]
    return torch.cat(feats, 0)


def evaluate(args):
    model, heads, device = load_models(args)
    results = {name: {} for name in heads}
    for d in args.datasets:
        samples = list_samples(args.data_root, d, args.split)
        y = np.array([s[1] for s in samples])
        feats = features_of(model, samples, args, device)
        for name, head in heads.items():
            with torch.inference_mode():
                logits = head(feats).float().cpu().numpy()
            r = evaluate_dataset(logits, y, GLOBAL_INDEX[d])
            r['n'] = len(samples)
            results[name][d] = r
        print('%-16s %5d images done' % (PAPER_NAME[d], len(samples)), flush=True)

    for name in heads:
        results[name]['AVERAGE'] = {m: float(np.mean([results[name][d][m] for d in args.datasets]))
                                    for m in METRICS}
    rows = args.datasets + ['AVERAGE']
    mean = {d: {m: float(np.mean([results[n][d][m] for n in heads])) for m in METRICS} for d in rows}
    std = {d: {m: float(np.std([results[n][d][m] for n in heads], ddof=1)) if len(heads) > 1 else 0.0
               for m in METRICS} for d in rows}

    def table(title, block):
        print('\n' + title)
        print('%-16s' % 'Dataset' + ''.join('%12s' % m.capitalize() for m in SHOWN))
        for d in rows:
            label = 'Average' if d == 'AVERAGE' else PAPER_NAME[d]
            print('%-16s' % label + ''.join('%12.2f' % (100 * block[d][m]) for m in SHOWN))

    for name in heads:
        table('Head %s (%s split, values in %%)' % (name, args.split), results[name])
    if len(heads) > 1:
        table('Mean over the %d heads (the values reported in the paper)' % len(heads), mean)
        print('\nAverage accuracy over the datasets: %.2f +- %.2f %% (mean and standard deviation over the heads)'
              % (100 * mean['AVERAGE']['accuracy'], 100 * std['AVERAGE']['accuracy']))

    out = {'split': args.split, 'heads': results, 'mean': mean, 'std': std}
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
        print('written:', args.out)
    if args.check:
        check(out, args)
    return out


def check(out, args):
    """Compare with the values stored in expected_results.json (test split, all datasets)."""
    with open(os.path.join(HERE, 'expected_results.json'), encoding='utf-8') as f:
        expected = json.load(f)
    worst = 0.0
    lines = []
    for name, res in out['heads'].items():
        ref = expected['heads'].get(name)
        if ref is None:
            continue
        for d in res:
            if d not in ref:
                continue
            for m in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
                diff = abs(100 * res[d][m] - 100 * ref[d][m])
                worst = max(worst, diff)
                if diff > 0.05:
                    lines.append('  %s %s %s: %.2f, expected %.2f' % (name, d, m, 100 * res[d][m], 100 * ref[d][m]))
    print('\nCheck against expected_results.json: largest difference %.3f percentage points' % worst)
    for line in lines[:20]:
        print(line)
    if worst <= 0.05:
        print('All values match.')


@torch.inference_mode()
def classify(args):
    model, heads, device = load_models(args)
    name, head = next(iter(heads.items()))
    if args.dataset:
        index = GLOBAL_INDEX[args.dataset]
        names = ['%s (%s)' % (LOCAL_CLASSES[args.dataset][i], GLOBAL_CLASSES[g]) for i, g in enumerate(index)]
    else:
        index = list(range(len(GLOBAL_CLASSES)))
        names = GLOBAL_CLASSES
    print('head: %s, classes: %d' % (name, len(index)))
    for path in args.images:
        x = load_image(path).unsqueeze(0).to(device)
        logits = head(backbone_features(model, x)).float().cpu().numpy()[:, index]
        prob = softmax(logits)[0]
        order = np.argsort(-prob)[:args.topk]
        print(path)
        for k in order:
            print('   %6.2f %%  %s' % (100 * prob[k], names[k]))


def save_full_model(args):
    model, heads, _ = load_models(args)
    name, head = next(iter(heads.items()))
    model.head.load_state_dict(head.state_dict())
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.save_full_model)
    print('written: %s (backbone and %s)' % (args.save_full_model, name))


if __name__ == '__main__':
    a = parse_args()
    if a.save_full_model:
        save_full_model(a)
    elif a.images:
        classify(a)
    else:
        evaluate(a)
