"""Prediction demo with OpenCV.

Usage: python scripts/predict.py path/to/leaf.jpg [--model custom|baseline]
       python scripts/predict.py --demo        (grid of random held-out test images)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

from src.data.acquire import square_resize  # noqa: E402
from src.training.train import load_manifest  # noqa: E402
from src.utils.common import load_config, project_path  # noqa: E402
from src.utils.plotting import CRITICAL, GOOD, plt, save  # noqa: E402


def prepare(path: Path, size: int, crop: int) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image {path}")
    img = square_resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)
    o = (size - crop) // 2
    return img[o:o + crop, o:o + crop].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--model", default="custom", choices=["custom", "baseline"])
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    classes = cfg["classes"]
    model = keras.models.load_model(project_path("models", f"vgg16_{a.model}.keras"))
    size, crop = cfg["data"]["image_size"], cfg["data"]["crop_size"]
    if a.image:
        x = prepare(Path(a.image), size, crop)[None]
        p = np.asarray(model(x, training=False)["probs"])[0]
        for i in np.argsort(-p):
            print(f"{classes[i]:<16} {p[i]:.4f}")
        return
    te = load_manifest(cfg).query("split == 'test'").sample(12, random_state=3)
    proc = project_path(cfg["data"]["processed_dir"])
    X = np.stack([prepare(proc / r, size, crop) for r in te.path])
    P = np.asarray(model(X, training=False)["probs"])
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    for ax, img, p, (_, r) in zip(axes.ravel(), X, P, te.iterrows()):
        k = int(p.argmax())
        ok = classes[k] == r.label
        ax.imshow(img.astype(np.uint8))
        ax.axis("off")
        ax.set_title(f"true: {r.label.replace('_', ' ')}\npred: {classes[k].replace('_', ' ')} ({p[k]:.2f})",
                     fontsize=9, color=GOOD if ok else CRITICAL)
    fig.suptitle(f"{a.model} model predictions on held-out test leaves")
    out = save(fig, project_path("results", "graphs", f"prediction_demo_{a.model}.png"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
