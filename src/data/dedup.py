"""Near-duplicate grouping with a rotation/flip-invariant difference hash (dHash).

PlantVillage contains re-saved and re-oriented copies of the same photo ("... copy.jpg", "... copy 2.jpg").
Copies sharing an original photo ID are grouped by name. This module additionally groups images whose 64-bit
dHash matches within `max_distance` bits under any of the 8 dihedral transforms (rotations/flips). Grouped images
always land in the same split, preventing train/test leakage through near-identical images.
"""
from __future__ import annotations

import cv2
import numpy as np


def dhash(gray: np.ndarray) -> np.uint64:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def dihedral_hashes(gray: np.ndarray) -> np.ndarray:
    out = []
    for k in range(4):
        r = np.rot90(gray, k)
        out += [dhash(r), dhash(np.fliplr(r))]
    return np.array(out, dtype=np.uint64)


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def near_duplicate_groups(hashes: np.ndarray, max_distance: int = 4, initial_groups=None) -> np.ndarray:
    """hashes [n, 8] (8 dihedral variants per image). Images in the same `initial_groups` label are unioned first.
    Returns a connected-component id per image (union-find)."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if initial_groups is not None:
        first = {}
        for i, g in enumerate(initial_groups):
            if g in first:
                ri, rj = find(first[g]), find(i)
                if ri != rj:
                    parent[rj] = ri
            else:
                first[g] = i

    canon = hashes[:, 0]
    for i in range(n - 1):
        # distance between image i (orientation 0) and every later image under all 8 orientations
        d = _popcount64(np.bitwise_xor(hashes[i + 1:], canon[i])).min(axis=1)
        for j in np.nonzero(d <= max_distance)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])
