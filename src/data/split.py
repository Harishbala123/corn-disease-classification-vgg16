"""Stratified, group-aware train/val/test split (no leaf group appears in two splits)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_split(man: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.Series:
    """Greedy stratified group split: groups are shuffled per class and assigned to the split that is furthest
    below its target share of that class's images."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=man.index, dtype=object)
    targets = {"test": test_fraction, "val": val_fraction, "train": 1 - val_fraction - test_fraction}
    # a group's class = its majority label (groups are single-class in practice)
    g_label = man.groupby("group")["label"].agg(lambda s: s.mode()[0])
    g_size = man.groupby("group").size()
    for cls in sorted(g_label.unique()):
        groups = g_label[g_label == cls].index.to_numpy().copy()   # pandas 3 returns read-only arrays
        rng.shuffle(groups)
        total = g_size[groups].sum()
        filled = {k: 0 for k in targets}
        for g in groups:
            k = min(targets, key=lambda s: filled[s] / total - targets[s])
            filled[k] += g_size[g]
            split[man.group == g] = k
    return split
