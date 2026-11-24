import numpy as np
import pandas as pd
import pytest

from src.data.acquire import _pv_original_id, square_resize
from src.data.split import group_split
from src.training.losses import HardNegativeSupCon, RewardRegretCE, WeightedCE, reward_matrix

CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]


def test_reward_matrix():
    R = reward_matrix(CLASSES, ["Blight", "Gray_Leaf_Spot"], 1, -1, -4)
    assert np.all(np.diag(R) == 1) and R[0, 2] == R[2, 0] == -4 and R[0, 1] == -1


def test_pv_original_id_strips_copy_suffix():
    assert _pv_original_id("0a1b___R.S_HL 8315 copy 2.jpg") == "R.S_HL 8315"
    assert _pv_original_id("0a1b___RS_GLSp 4655.JPG") == "RS_GLSp 4655"


def test_square_resize_keeps_center():
    img = np.zeros((100, 300, 3), np.uint8)
    img[:, 100:200] = 255
    out = square_resize(img, 64)
    assert out.shape == (64, 64, 3) and out[32, 32].mean() > 200


def test_near_duplicates_grouped_under_rotation_and_names():
    from src.data.dedup import dihedral_hashes, near_duplicate_groups
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype(np.uint8)
    b = (rng.random((64, 64)) * 255).astype(np.uint8)
    c = (rng.random((64, 64)) * 255).astype(np.uint8)
    H = np.stack([dihedral_hashes(x) for x in (a, np.rot90(a), np.fliplr(a), b, c)])
    g = near_duplicate_groups(H, 4, initial_groups=["x", "y", "z", "w", "w"])
    assert g[0] == g[1] == g[2]          # rotated / flipped copy of a
    assert g[3] == g[4] != g[0]          # b and c share a name group


def test_group_split_has_no_leakage_and_is_stratified():
    rng = np.random.default_rng(0)
    rows = [{"group": f"g{c}_{g}", "label": c} for c in CLASSES for g in range(60) for _ in range(rng.integers(1, 4))]
    man = pd.DataFrame(rows)
    man["split"] = group_split(man, 0.15, 0.15, 1)
    assert man.groupby("group").split.nunique().max() == 1
    share = man.groupby("label").split.value_counts(normalize=True).unstack()
    assert share["test"].between(0.08, 0.22).all() and share["train"].between(0.6, 0.8).all()


tf = pytest.importorskip("tensorflow")


def test_reward_loss_penalises_critical_confusion_more():
    R = reward_matrix(CLASSES, ["Blight", "Gray_Leaf_Spot"], 1, -1, -4)
    loss = RewardRegretCE([1, 1, 1, 1], R, lambda_reward=1.0)
    y = np.array([[1, 0, 0, 0]], np.float32)                       # true Blight
    to_gls = np.array([[0.3, 0.0, 0.7, 0.0]], np.float32)           # mass on Gray Leaf Spot (critical)
    to_rust = np.array([[0.3, 0.7, 0.0, 0.0]], np.float32)          # same mass on Common Rust (ordinary)
    assert float(loss(y, to_gls)) > float(loss(y, to_rust))
    # with equal probabilities the CE part is identical, so the difference is exactly the regret term
    ce = WeightedCE([1, 1, 1, 1])
    assert float(ce(y, to_gls)) == pytest.approx(float(ce(y, to_rust)))
    assert float(loss(y, to_gls)) - float(loss(y, to_rust)) == pytest.approx(0.7 * (-1 - -4), rel=1e-5)


def test_regret_zero_for_perfect_prediction():
    R = reward_matrix(CLASSES, ["Blight", "Gray_Leaf_Spot"])
    loss = RewardRegretCE([1, 1, 1, 1], R, lambda_reward=1.0)
    y = np.eye(4, dtype=np.float32)
    assert float(loss(y, y)) == pytest.approx(0.0, abs=1e-5)


def test_supcon_prefers_separated_embeddings_and_hard_negative_weight():
    y = np.array([0, 0, 2, 2], np.float32)
    good = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], np.float32)
    bad = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32)
    l = HardNegativeSupCon(0.5, (0, 2), 1.0)
    assert float(l(y, good)) < float(l(y, bad))
    mixed = np.array([[1, 0], [0.8, 0.6], [0.6, 0.8], [0, 1]], np.float32)
    l_hard = HardNegativeSupCon(0.5, (0, 2), 3.0)
    assert float(l_hard(y, mixed)) > float(l(y, mixed))   # Blight/GLS similarity is penalised more
