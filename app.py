"""Interactive demo: baseline VGG16 vs custom-loss VGG16 on corn leaf images.

Run:  streamlit run app.py        (or double-click run_demo.bat)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import keras
import numpy as np
import pandas as pd
import streamlit as st

import src.models.vgg16_model  # noqa: F401  (registers custom layer for loading)
import src.training.losses  # noqa: F401  (registers custom losses for loading)
from src.data.acquire import square_resize
from src.utils.common import load_config, project_path
from src.utils.plotting import SERIES, plt

st.set_page_config(page_title="Corn Leaf Disease Demo", layout="wide")
cfg = load_config()
CLASSES = cfg["classes"]
RELEASE = "https://github.com/Harishbala123/corn-disease-classification-vgg16/releases/download/v1.0/"
MODEL_FILES = {"Baseline (cross-entropy)": "vgg16_baseline.keras", "Custom loss (RL-inspired + contrastive)": "vgg16_custom.keras"}


@st.cache_resource
def load_models():
    return {k: keras.models.load_model(project_path("models", f)) for k, f in MODEL_FILES.items()}


def prepare(rgb: np.ndarray) -> np.ndarray:
    size, crop = cfg["data"]["image_size"], cfg["data"]["crop_size"]
    img = square_resize(rgb, size)
    o = (size - crop) // 2
    return img[o:o + crop, o:o + crop].astype(np.float32)


def decode(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.title("Corn Leaf Disease Classification: VGG16 with an RL-Inspired Loss")
st.caption("Four classes: Blight, Common Rust, Gray Leaf Spot, Healthy. Compare the baseline with the model trained "
           "to avoid the costly Blight vs Gray Leaf Spot confusion.")

missing = [f for f in MODEL_FILES.values() if not project_path("models", f).exists()]
if missing:
    st.warning(f"Trained models not found in models/: {', '.join(missing)} (~57 MB each, GitHub release v1.0).")
    if st.button("Download models from the GitHub release"):
        for f in missing:
            with st.spinner(f"Downloading {f}..."):
                urllib.request.urlretrieve(RELEASE + f, project_path("models", f))
        st.rerun()
    st.stop()

models = load_models()
tab_demo, tab_results, tab_how = st.tabs(["Try it", "Experiment results", "How the loss works"])

with tab_demo:
    samples = sorted(project_path("samples").glob("*.jpg"))
    src = st.radio("Image source", ["Sample test leaf", "Upload your own"], horizontal=True)
    rgb, truth = None, None
    if src == "Sample test leaf":
        pick = st.selectbox("Sample (held-out test set)", samples, format_func=lambda p: p.stem.replace("_", " "))
        rgb = decode(Path(pick).read_bytes())
        truth = next((c for c in CLASSES if Path(pick).stem.lower().startswith(c.lower())), None)
    else:
        up = st.file_uploader("Leaf photo (jpg / png)", type=["jpg", "jpeg", "png"])
        if up:
            rgb = decode(up.getvalue())
            if rgb is None:
                st.error("Could not read that image.")
    if rgb is not None:
        x = prepare(rgb)
        left, right = st.columns([1, 2])
        left.image(x.astype(np.uint8), caption="Model input (224 × 224 centre crop)")
        if truth:
            left.write(f"True label: **{truth.replace('_', ' ')}**")
        probs = {k: np.asarray(m(x[None], training=False)["probs"])[0] for k, m in models.items()}
        fig, ax = plt.subplots(figsize=(7, 3.4))
        y = np.arange(len(CLASSES))
        for i, (k, p) in enumerate(probs.items()):
            ax.barh(y + (i - 0.5) * 0.38, p, 0.36, color=SERIES[i], label=k)
        ax.set_yticks(y, [c.replace("_", " ") for c in CLASSES])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.legend(loc="lower right", fontsize=8)
        right.pyplot(fig)
        for k, p in probs.items():
            right.write(f"**{k}:** {CLASSES[int(p.argmax())].replace('_', ' ')} ({p.max():.1%})")

with tab_results:
    res = project_path("results")
    st.dataframe(pd.read_csv(res / "metrics" / "model_comparison.csv").round(4))
    comp = json.loads((res / "metrics" / "evaluation_metrics.json").read_text())["comparison"]
    lo, hi = comp["critical_error_reduction_95ci_pct"]
    st.write(f"Critical Blight/Gray Leaf Spot errors: **{comp['critical_errors_baseline']} → "
             f"{comp['critical_errors_custom']}** ({comp['critical_error_reduction_pct']:.1f}% fewer; "
             f"bootstrap 95% CI {lo:.1f}–{hi:.1f}%).")
    c1, c2 = st.columns(2)
    c1.image(str(res / "graphs" / "confusion_matrix_baseline.png"))
    c2.image(str(res / "graphs" / "confusion_matrix_custom.png"))
    st.image(str(res / "graphs" / "embedding_tsne.png"))
    st.image(str(res / "graphs" / "per_class_f1.png"))
    st.dataframe(pd.read_csv(res / "metrics" / "accuracy_by_source.csv").round(4))

with tab_how:
    st.markdown("""
**Total loss = weighted cross-entropy + expected regret + 0.5 × supervised contrastive loss**

- **Expected regret (RL-inspired):** the softmax output is treated as a policy choosing a diagnosis. A reward
  matrix gives +1 for correct, −1 for an ordinary error, and **−4 for Blight ↔ Gray Leaf Spot**. The loss maximises
  expected reward, computed exactly over the 4 actions; its gradient is the policy gradient with a baseline.
  It is *inspired by* RL: there is no environment, exploration or sequential decision, so mathematically it is
  cost-sensitive learning.
- **Hard-negative supervised contrastive loss:** pulls same-class embeddings together and gives Blight/Gray Leaf
  Spot pairs 3× weight as negatives, separating exactly the two confusable classes (see the t-SNE plot).
- **Fair comparison:** both models share the data split, augmentation, class weights, schedule and seed.
""")
    st.code(json.dumps(cfg["custom_loss"], indent=2), language="json")
