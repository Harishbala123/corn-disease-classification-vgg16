"""Evaluate baseline vs custom-loss VGG16 on the same held-out test set."""
from __future__ import annotations

import keras
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, silhouette_score)

from src.data.pipeline import full_paths, make_dataset
from src.training.train import VARIANTS, load_manifest
from src.utils.common import get_logger, project_path, save_json
from src.utils.plotting import INK_2, SERIES, plt, save

log = get_logger(__name__)


def predict(model, ds):
    probs, embs = [], []
    for x, _ in ds:
        out = model(x, training=False)
        probs.append(np.asarray(out["probs"]))
        embs.append(np.asarray(out["embedding"]))
    return np.concatenate(probs), np.concatenate(embs)


def critical_errors(y, pred, a, b) -> int:
    return int(np.sum((y == a) & (pred == b)) + np.sum((y == b) & (pred == a)))


def plot_cm(cm, classes, path, title):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.6 else "#0b0b0b")
    labels = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=20)
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    save(fig, path)


def evaluate(cfg: dict, n_boot: int = 5000) -> dict:
    classes = cfg["classes"]
    a, b = (classes.index(c) for c in cfg["critical_pair"])
    man = load_manifest(cfg)
    te = man[man.split == "test"].reset_index(drop=True)
    proc = project_path(cfg["data"]["processed_dir"])
    ds = make_dataset(full_paths(proc, te.path), te.y.to_numpy(), len(classes), 64, cfg["data"]["crop_size"],
                      False, 0)
    y = te.y.to_numpy()
    res = project_path("results")
    results, preds, embs = {}, {}, {}
    for v in VARIANTS:
        model = keras.models.load_model(project_path("models", f"vgg16_{v}.keras"))
        prob, emb = predict(model, ds)
        pred = prob.argmax(1)
        preds[v], embs[v] = pred, emb
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
        cm = confusion_matrix(y, pred, labels=range(len(classes)))
        plot_cm(cm, classes, res / "graphs" / f"confusion_matrix_{v}.png",
                f"{'Baseline' if v == 'baseline' else 'Custom-loss'} VGG16: test confusion matrix")
        rep = classification_report(y, pred, target_names=classes, digits=4)
        (res / "reports" / f"classification_report_{v}.txt").write_text(rep)
        pd.DataFrame(classification_report(y, pred, target_names=classes, output_dict=True)).T.to_csv(
            res / "metrics" / f"classification_report_{v}.csv")
        crit_mask = np.isin(y, [a, b])
        results[v] = {
            "accuracy": accuracy_score(y, pred),
            "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(classes)},
            "critical_errors": critical_errors(y, pred, a, b),
            "blight_to_gls": int(cm[a, b]), "gls_to_blight": int(cm[b, a]),
            "blight_gls_silhouette": float(silhouette_score(emb[crit_mask], y[crit_mask])),
            "confusion_matrix": cm.tolist(),
        }
        log.info("%s: acc %.4f  macro-F1 %.4f  critical errors %d", v, results[v]["accuracy"], results[v]["macro_f1"],
                 results[v]["critical_errors"])

    # Per-image predictions and accuracy by image source (lab vs field photos)
    pred_df = te[["path", "label", "source"]].assign(pred_baseline=[classes[i] for i in preds["baseline"]],
                                                     pred_custom=[classes[i] for i in preds["custom"]])
    pred_df.to_csv(res / "metrics" / "test_predictions.csv", index=False)
    by_source = pred_df.groupby("source").apply(
        lambda d: pd.Series({"images": len(d), "acc_baseline": (d.pred_baseline == d.label).mean(),
                             "acc_custom": (d.pred_custom == d.label).mean()}), include_groups=False)
    by_source.to_csv(res / "metrics" / "accuracy_by_source.csv")
    log.info("Accuracy by source:\n%s", by_source)

    # Paired bootstrap over test images for the change in critical errors and accuracy
    rng = np.random.default_rng(cfg["seed"])
    base_err = ((y == a) & (preds["baseline"] == b)) | ((y == b) & (preds["baseline"] == a))
    cust_err = ((y == a) & (preds["custom"] == b)) | ((y == b) & (preds["custom"] == a))
    base_ok, cust_ok = preds["baseline"] == y, preds["custom"] == y
    red, dacc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        be, ce = base_err[i].sum(), cust_err[i].sum()
        if be > 0:
            red.append((be - ce) / be)
        dacc.append(cust_ok[i].mean() - base_ok[i].mean())
    be, ce = results["baseline"]["critical_errors"], results["custom"]["critical_errors"]
    comparison = {
        "n_test_images": int(len(y)),
        "n_test_blight_or_gls": int(np.isin(y, [a, b]).sum()),
        "critical_errors_baseline": be, "critical_errors_custom": ce,
        "critical_error_reduction_pct": float(100 * (be - ce) / be) if be else None,
        "critical_error_reduction_95ci_pct": [float(100 * np.percentile(red, 2.5)), float(100 * np.percentile(red, 97.5))],
        "accuracy_baseline": results["baseline"]["accuracy"], "accuracy_custom": results["custom"]["accuracy"],
        "accuracy_difference_95ci": [float(np.percentile(dacc, 2.5)), float(np.percentile(dacc, 97.5))],
    }
    save_json({"models": results, "comparison": comparison}, res / "metrics" / "evaluation_metrics.json")
    rows = [{"model": v, "accuracy": results[v]["accuracy"], "macro_precision": results[v]["macro_precision"],
             "macro_recall": results[v]["macro_recall"], "macro_f1": results[v]["macro_f1"],
             "blight_to_gls": results[v]["blight_to_gls"], "gls_to_blight": results[v]["gls_to_blight"],
             "critical_errors": results[v]["critical_errors"],
             "blight_gls_embedding_silhouette": results[v]["blight_gls_silhouette"]} for v in VARIANTS]
    pd.DataFrame(rows).to_csv(res / "metrics" / "model_comparison.csv", index=False)
    pd.DataFrame([{"model": v, "class": c, **results[v]["per_class"][c]} for v in VARIANTS for c in classes]).to_csv(
        res / "metrics" / "per_class_metrics.csv", index=False)
    log.info("Comparison: %s", comparison)
    plot_curves(res)
    plot_per_class(results, classes, res / "graphs" / "per_class_f1.png")
    plot_embeddings(embs, y, classes, res / "graphs" / "embedding_tsne.png", cfg["seed"])
    return {"models": results, "comparison": comparison}


def plot_curves(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, v in enumerate(VARIANTS):
        p = res / "metrics" / f"history_{v}.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        e = np.arange(1, len(h) + 1)
        lab = "Baseline" if v == "baseline" else "Custom loss"
        axes[0].plot(e, h["probs_accuracy"], color=SERIES[i], ls="--", label=f"{lab} train")
        axes[0].plot(e, h["val_probs_accuracy"], color=SERIES[i], label=f"{lab} val")
        axes[1].plot(e, h["loss"], color=SERIES[i], ls="--", label=f"{lab} train")
        axes[1].plot(e, h["val_loss"], color=SERIES[i], label=f"{lab} val")
        n_head = int((h.stage == "head").sum())
        for ax in axes:
            ax.axvline(n_head + 0.5, color=INK_2, lw=0.8, ls=":")
    axes[0].set_title("Accuracy (dotted line: start of block5 fine-tuning)")
    axes[1].set_title("Total loss (objectives differ, so compare within a model)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    save(fig, res / "graphs" / "training_curves.png")


def plot_per_class(results, classes, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(classes))
    for i, v in enumerate(VARIANTS):
        f1 = [results[v]["per_class"][c]["f1"] for c in classes]
        ax.bar(x + (i - 0.5) * 0.38, f1, 0.36, color=SERIES[i], label="Baseline" if v == "baseline" else "Custom loss")
        for xi, val in zip(x, f1):
            ax.text(xi + (i - 0.5) * 0.38, val + 0.005, f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [c.replace("_", " ") for c in classes])
    lo = min(results[v]["per_class"][c]["f1"] for v in VARIANTS for c in classes)
    ax.set_ylim(max(0, lo - 0.08), 1.02)
    ax.set_ylabel("F1-score (test)")
    ax.set_title("Per-class F1: baseline vs custom loss")
    ax.legend()
    save(fig, path)


def plot_embeddings(embs, y, classes, path, seed):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, v in zip(axes, VARIANTS):
        z = TSNE(2, random_state=seed, init="pca", perplexity=30).fit_transform(embs[v])
        for i, c in enumerate(classes):
            m = y == i
            ax.scatter(z[m, 0], z[m, 1], s=8, color=SERIES[i], label=c.replace("_", " "), alpha=0.8)
        ax.set_title(f"{'Baseline' if v == 'baseline' else 'Custom loss'}: t-SNE of test embeddings")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(markerscale=2)
    save(fig, path)
