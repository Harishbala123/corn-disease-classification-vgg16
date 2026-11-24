# Trained models

The two trained models (~60 MB each, inference-only `.keras` files without optimizer state) are too large to
keep comfortably in Git, so they are attached to the GitHub Release **v1.0**:

| File | Description |
|---|---|
| `vgg16_baseline.keras` | VGG16 + head, class-weighted cross-entropy |
| `vgg16_custom.keras` | Same architecture, trained with CE + expected-regret (RL-inspired) + hard-negative SupCon loss |

Download into this folder:
```powershell
gh release download v1.0 --repo Harishbala123/corn-disease-classification-vgg16 --dir models
```
or from the Releases page in your browser. Alternatively, retrain with `python scripts/train.py --variant baseline|custom`.

The custom loss classes are registered Keras serialisables (`src/training/losses.py`), and `VGGPreprocess` lives in
`src/models/vgg16_model.py`. Import these modules before `keras.models.load_model(...)`. The scripts already do this.
