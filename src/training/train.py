"""Train the baseline and the custom-loss VGG16 models with an identical schedule, data and seed."""
from __future__ import annotations

import time

import keras
import numpy as np
import pandas as pd

from src.data.pipeline import full_paths, make_dataset
from src.models.vgg16_model import build_model, set_trainable
from src.training.losses import HardNegativeSupCon, RewardRegretCE, WeightedCE, reward_matrix
from src.utils.common import get_logger, project_path, save_json, set_seed
from src.utils.export import save_inference_model

log = get_logger(__name__)
VARIANTS = ("baseline", "custom")


def load_manifest(cfg) -> pd.DataFrame:
    man = pd.read_csv(project_path(cfg["data"]["processed_dir"], "splits.csv"))
    man["y"] = man.label.map({c: i for i, c in enumerate(cfg["classes"])})
    return man


def class_weights(y: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n).astype(float)
    return counts.sum() / (n * counts)


def make_losses(cfg, variant: str, cw: np.ndarray) -> tuple[dict, dict]:
    classes, cl = cfg["classes"], cfg["custom_loss"]
    if variant == "baseline":
        return {"probs": WeightedCE(cw), "embedding": None}, {"probs": 1.0, "embedding": 0.0}
    R = reward_matrix(classes, cfg["critical_pair"], cl["reward_correct"], cl["penalty_wrong"], cl["penalty_critical"])
    crit = [classes.index(c) for c in cfg["critical_pair"]]
    return ({"probs": RewardRegretCE(cw, R, cl["lambda_reward"]),
             "embedding": HardNegativeSupCon(cl["temperature"], crit, cl["hard_negative_weight"])},
            {"probs": 1.0, "embedding": cl["lambda_contrastive"]})


def train(cfg: dict, variant: str, epochs_override: dict | None = None) -> dict:
    assert variant in VARIANTS
    set_seed(cfg["seed"])
    man = load_manifest(cfg)
    proc = project_path(cfg["data"]["processed_dir"])
    n = len(cfg["classes"])
    tr, va = man[man.split == "train"], man[man.split == "val"]
    tc = {**cfg["training"], **(epochs_override or {})}
    ds_tr = make_dataset(full_paths(proc, tr.path), tr.y.to_numpy(), n, tc["batch_size"], cfg["data"]["crop_size"],
                         True, cfg["seed"])
    ds_va = make_dataset(full_paths(proc, va.path), va.y.to_numpy(), n, tc["batch_size"], cfg["data"]["crop_size"],
                         False, cfg["seed"])
    cw = class_weights(tr.y.to_numpy(), n)
    losses, loss_weights = make_losses(cfg, variant, cw)
    m = cfg["model"]
    model = build_model(n, cfg["data"]["crop_size"], m["embedding_dim"], m["dense_units"], m["dropout"])
    out = project_path("models", f"vgg16_{variant}.keras")

    def stage(name, epochs, lr):
        set_trainable(model, name)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, loss_weights=loss_weights,
                      metrics={"probs": ["accuracy"]})
        log.info("[%s] stage %s: %d trainable weights tensors", variant, name, len(model.trainable_weights))
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=tc["patience"], restore_best_weights=True)]
        h = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=cbs, verbose=2)
        return pd.DataFrame(h.history).assign(stage=name)

    t0 = time.time()
    h1 = stage("head", tc["head_epochs"], tc["head_lr"])
    w1, best1 = model.get_weights(), float(h1["val_loss"].min())
    h2 = stage("finetune", tc["finetune_epochs"], tc["finetune_lr"])
    if float(h2["val_loss"].min()) >= best1:      # keep fine-tuning only if it beat stage 1 on validation
        model.set_weights(w1)
        log.info("[%s] fine-tuning did not improve validation loss: keeping stage-1 weights", variant)
    hist = pd.concat([h1, h2], ignore_index=True)
    save_inference_model(model, out)   # weights only, no optimizer state
    hist.index.name = "epoch"
    hist.to_csv(project_path("results", "metrics", f"history_{variant}.csv"))
    info = {"variant": variant, "seconds": round(time.time() - t0, 1), "class_weights": cw.tolist(),
            "n_train": int(len(tr)), "n_val": int(len(va)), "epochs": int(len(hist))}
    save_json(info, project_path("results", "metrics", f"training_info_{variant}.json"))
    log.info("[%s] done in %.0fs", variant, info["seconds"])
    return info
