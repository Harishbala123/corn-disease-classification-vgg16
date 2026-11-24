"""Download sources, validate images with OpenCV, de-duplicate, split by leaf group, and summarise.

Usage: python scripts/acquire_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.acquire import build_manifest, download_sources  # noqa: E402
from src.data.split import group_split  # noqa: E402
from src.utils.common import get_logger, load_config, project_path, save_json  # noqa: E402
from src.utils.plotting import SERIES, plt, save  # noqa: E402

log = get_logger("acquire")

if __name__ == "__main__":
    cfg = load_config()
    raw, proc = project_path(cfg["data"]["raw_dir"]), project_path(cfg["data"]["processed_dir"])
    download_sources(raw)
    man = build_manifest(raw, proc, cfg["data"]["image_size"], cfg["data"]["min_side"])
    man["split"] = group_split(man, cfg["data"]["val_fraction"], cfg["data"]["test_fraction"], cfg["seed"])
    man.to_csv(proc / "splits.csv", index=False)

    # leakage check: no leaf group in more than one split
    assert man.groupby("group").split.nunique().max() == 1
    dist = man.pivot_table(index="label", columns="split", values="path", aggfunc="count").fillna(0).astype(int)
    dist = dist[["train", "val", "test"]]
    dist["total"] = dist.sum(1)
    src = man.pivot_table(index="label", columns="source", values="path", aggfunc="count").fillna(0).astype(int)
    summary = dist.join(src)
    summary.to_csv(project_path("results", "metrics", "dataset_summary.csv"))
    rej = pd.read_csv(proc / "rejected.csv")
    save_json({"kept": int(len(man)), "rejected": rej.reason.value_counts().to_dict(),
               "plantvillage_leaf_map_coverage": float(man[man.source == "PlantVillage"].leaf_map_hit.mean()),
               "n_groups": int(man.group.nunique())}, project_path("results", "metrics", "dataset_validation.json"))
    print(summary)

    fig, ax = plt.subplots(figsize=(8, 4))
    classes = cfg["classes"]
    x = np.arange(len(classes))
    for i, s in enumerate(["train", "val", "test"]):
        ax.bar(x + (i - 1) * 0.27, dist.loc[classes, s], 0.25, color=SERIES[i], label=s)
    ax.set_xticks(x, [c.replace("_", " ") for c in classes])
    ax.set_ylabel("Images")
    ax.set_title("Class distribution per split (grouped by physical leaf)")
    ax.legend()
    save(fig, project_path("results", "graphs", "class_distribution.png"))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    rng = np.random.default_rng(0)
    for j, c in enumerate(classes):
        for i, srcname in enumerate(["PlantVillage", "PlantDoc"]):
            ax = axes[i, j]
            ax.axis("off")
            pool = man[(man.label == c) & (man.source == srcname)]
            if len(pool):
                r = pool.iloc[rng.integers(len(pool))]
                img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(proc / r.path), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"{c.replace('_', ' ')}\n({srcname})", fontsize=9)
    fig.suptitle("Dataset samples: lab images (PlantVillage, top) vs field images (PlantDoc, bottom)")
    save(fig, project_path("results", "graphs", "dataset_samples.png"))
