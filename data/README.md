# Data: corn leaf disease images

Images are **not committed**. `python scripts/acquire_data.py` downloads, validates, de-duplicates and splits them.

## Sources and licences
| Source | Images used | Licence | Citation |
|---|---|---|---|
| PlantVillage (lab images, uniform background) | 4 corn classes | CC BY-SA 3.0 | Hughes & Salathé, arXiv:1511.08060 (2015); Mohanty, Hughes & Salathé, *Front. Plant Sci.* 7:1419 (2016). github.com/spMohanty/PlantVillage-Dataset |
| PlantDoc (field images, cluttered backgrounds) | 3 corn disease classes | CC BY 4.0 | Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", CODS-COMAD 2020. github.com/pratikkayal/PlantDoc-Dataset |

The Kaggle "Corn or Maize Leaf Disease Dataset" (Ghose, 2020) was assembled from these same two sources.
Rebuilding it from the originals gives clear licensing, needs no Kaggle account, and keeps the PlantVillage
*leaf-grouping* metadata, which is required for a leakage-free split.

## Class mapping
| Class | PlantVillage folder | PlantDoc folder |
|---|---|---|
| Blight | `Corn_(maize)___Northern_Leaf_Blight` | `Corn leaf blight` |
| Common_Rust | `Corn_(maize)___Common_rust_` | `Corn rust leaf` |
| Gray_Leaf_Spot | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn Gray leaf spot` |
| Healthy | `Corn_(maize)___healthy` | (none) |

## Validation and splitting (performed by the script)
1. Every file is decoded with OpenCV. Unreadable files and files with a side < 64 px are rejected.
2. Images are resized (short side 256) and centre-cropped to 256×256.
3. Exact pixel duplicates (after resizing) are removed: the first copy is kept, and later copies are logged in `data/processed/rejected.csv`, including those with a conflicting label.
4. Grouping: PlantVillage's `leaf_grouping/leaf-map.json` is consulted, but it contains **no corn entries**. Groups are therefore built from the original photo ID (merging "... copy.jpg" files) plus a rotation/flip-invariant perceptual hash (`src/data/dedup.py`) that merges near-duplicates. PlantDoc images are their own group unless they hash-match another image.
5. Each class is split ~70/15/15 by group. The script asserts that no group appears in two splits.

Note: one PlantDoc file name collided after sanitising (Windows-invalid characters), so 375 of the 376 PlantDoc corn files were used.

Outputs: `data/processed/images/<class>/<md5>.jpg`, `data/processed/splits.csv`, `results/metrics/dataset_summary.csv`.
