'''
CIFAR-10 one class against the rest: trains each network and stores its logit on every
mask of a 4x4 grid of patches, for query images of each class. Writes results/raw/images/.
Runs on the GPU, in the environment of the imbalanced-explanations repository.

usage: uv run --project ../../../../imbalanced-explanations/imbalanced-explanations \\
           python run_images.py [--ratios 1 10 100] [--arms plain] [--seeds 2] [--pretrained]
       --pictures stores the query images themselves instead, for figures
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from data_loaders.imbalance import Problem
from oversample_curriculum.images import CLASSES, cifar10, load
from imbalanced_explanations.images import _normalise, train_net

RAW = Path(__file__).resolve().parent/'results'/'raw'/'images'
GRID = 4            # patches per side
QUERIES = 15        # test images per class
EPOCHS = {False: 50, True: 10}   # from scratch, fine-tuned
# this run's name for each training arm of imbalanced_explanations.images
ARMS = {'plain': 'imbalanced', 'weighted': 'class_weighted', 'logit adjusted': 'logit_adjusted'}
BATCH = 8192
BALANCED = 4000     # rows per class of the balanced development set; the rest are validation


@torch.no_grad()
def logits(net, X, batch=BATCH):
    '''float32 logit of the minority class for uint8 images'''
    out = [net.logits(_normalise(x).contiguous(memory_format=torch.channels_last)).float().squeeze(1)
           for x in X.split(batch)]
    return torch.cat(out)


def all_masks(device):
    '''(2**n, n) masks of the n patches; bit j of the row index is patch j'''
    n = GRID*GRID
    index = torch.arange(2**n, device=device)
    return ((index[:, None] >> torch.arange(n, device=device)) & 1).bool()


@torch.no_grad()
def enumerate_masks(net, image, masks, batch=BATCH):
    '''the logit of `image` (3, 32, 32) under every mask, hidden patches set to their mean colour'''
    size = image.shape[-1]//GRID
    mean = F.interpolate(F.avg_pool2d(image[None].float(), size), scale_factor=size, mode='nearest')
    out = []
    for part in masks.split(batch):
        keep = F.interpolate(part.view(-1, 1, GRID, GRID).float(), scale_factor=size, mode='nearest')
        X = (image[None].float()*keep + mean*(1 - keep)).round().to(torch.uint8)
        out.append(logits(net, X, batch))
    return torch.cat(out)


def balanced(positive, seed):
    '''the class against the rest at 1:1 in every split'''
    X, y, X_test, y_test = cifar10()
    rng = np.random.default_rng(seed)
    pos, neg = (rng.permutation(np.flatnonzero(keep)) for keep in (y == positive, y != positive))
    dev = np.concatenate([neg[:BALANCED], pos[:BALANCED]])
    val = np.concatenate([neg[BALANCED:len(pos)], pos[BALANCED:]])
    pos_t, neg_t = np.flatnonzero(y_test == positive), np.flatnonzero(y_test != positive)
    test = np.concatenate([rng.choice(neg_t, len(pos_t), replace=False), pos_t])
    binary = lambda labels: (labels == positive).astype(int)
    return Problem(CLASSES[positive], X[dev], binary(y[dev]), X_test[test], binary(y_test[test]),
                   seed, X[val], binary(y[val]))


def problem_for(positive, seed, ratio):
    return balanced(positive, seed) if ratio == 1 else load(positive, seed, imbalance=ratio)


def queries(problem, positive, seed):
    '''test rows explained: QUERIES of each class, the same for every training arm'''
    rng = np.random.default_rng([seed, positive, 20261119])
    return np.concatenate([rng.choice(np.flatnonzero(problem.y_test == c), QUERIES, replace=False)
                           for c in (0, 1)])


def pictures(positive, seed, ratio):
    problem = problem_for(positive, seed, ratio)
    query = queries(problem, positive, seed)
    path = RAW.parent/'images_pictures'/f'{ratio}_{CLASSES[positive]}_{seed}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=problem.X_test[query], query=query, y_query=problem.y_test[query])


def run(positive, seed, ratio, arm, pretrained):
    name = f"{'pretrained' if pretrained else 'scratch'}_{ratio}_{CLASSES[positive]}_{seed}_{arm}"
    path = RAW/f'{name.replace(" ", "-")}.npz'
    if path.exists():
        return
    start = time.time()
    problem = problem_for(positive, seed, ratio)
    device = torch.device('cuda')
    net = train_net(problem, ARMS[arm], seed, EPOCHS[pretrained], pretrained=pretrained).float()
    split = {k: logits(net, torch.from_numpy(X).to(device)).cpu().numpy()
             for k, X in (('train', problem.X), ('val', problem.X_val), ('test', problem.X_test))}
    query = queries(problem, positive, seed)
    masks = all_masks(device)
    X_test = torch.from_numpy(problem.X_test).to(device)
    found = torch.stack([enumerate_masks(net, X_test[i], masks) for i in query]).cpu().numpy()
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez(path, masks=found, query=query, y_query=problem.y_test[query],
             train=split['train'], y_train=problem.y, val=split['val'], y_val=problem.y_val,
             test=split['test'], y_test=problem.y_test,
             network='pretrained' if pretrained else 'scratch', ratio=ratio,
             dataset=CLASSES[positive], seed=seed, arm=arm)
    print(f'{name}: {time.time() - start:.0f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratios', type=int, nargs='+', default=[1, 10, 100])
    parser.add_argument('--classes', type=int, nargs='+', default=list(range(len(CLASSES))))
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--arms', nargs='+', default=['plain'], choices=list(ARMS))
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pictures', action='store_true')
    args = parser.parse_args()
    for ratio in args.ratios:
        for arm in args.arms:
            if ratio == 1 and arm != 'plain':
                continue
            for positive in args.classes:
                for seed in range(args.seeds):
                    if args.pictures:
                        pictures(positive, seed, ratio)
                    else:
                        run(positive, seed, ratio, arm, args.pretrained)
