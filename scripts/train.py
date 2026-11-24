"""Train one variant.  Usage: python scripts/train.py --variant baseline|custom"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import train  # noqa: E402
from src.utils.common import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "custom"], required=True)
    ap.add_argument("--head-epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    a = ap.parse_args()
    over = {k: v for k, v in {"head_epochs": a.head_epochs, "finetune_epochs": a.finetune_epochs}.items() if v}
    train(load_config(), a.variant, over)
