# BirdCLEF+ 2026

Reproducible training pipeline for the BirdCLEF+ competition. Designed for Google Colab with the project hosted on Google Drive.

## Setup

1. Clone this repository into `BirdCLEF_Project/repro/` on Google Drive.
2. Open notebooks from Drive in Colab (GPU runtime).
3. Add `KAGGLE_API_TOKEN` in Colab secrets for notebook 02.

Notebook 02 downloads the BirdCLEF competition data and the Perch v2 ONNX model automatically. After a first run, optional embedding archives can live in `BirdCLEF_Project/` on Drive:

```
BirdCLEF_Project/
├── perch_v2_no_dft.onnx          # cached by notebook 02 (~394 MB)
├── embeddings_v2_archive.zip     # optional, after extraction
├── embeddings_v2_TTA_archive.zip   # optional, after extraction
└── repro/
    ├── birdclef/
    ├── data/metadata/
    ├── outputs/
    └── *.ipynb
```

During training, embeddings are unzipped to `/content/` for speed. Models and figures are saved under `repro/outputs/`.

**Git note:** PyTorch checkpoints (`.pth`) and the Perch ONNX file are not in this repo. Checkpoints are ~15 MB each; Perch is ~394 MB. After training, keep them on Drive. Your MoE ONNX exports (~33 KB each) and PNG figures are small enough to commit if you want them in the repo.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_exploration.ipynb` | Metadata EDA |
| 02 | `02_download_and_extract_embeddings.ipynb` | Download data + Perch ONNX, extract embeddings |
| 03 | `03_train_ce_baseline.ipynb` | Model A — CE baseline (LB 0.773) |
| 04 | `04_train_mixup.ipynb` | Model B — embedding mixup (LB 0.739) |
| 05 | `05_train_tta_baseline.ipynb` | Model C — TTA + CE (LB 0.798) |
| 06 | `06_train_focal_gamma2.ipynb` | Focal loss γ=2, 5-fold |
| 07 | `07_train_best_model.ipynb` | Final model (γ=1, wd=1e-4, TTA) |
| 08 | `08_sweep_gamma.ipynb` | Focal γ sweep (proxy fold) |
| 09 | `09_sweep_weight_decay.ipynb` | Weight decay sweep |
| 10 | `10_sweep_learning_rate.ipynb` | Learning rate sweep |
| 11 | `11_sweep_context.ipynb` | Context window sweep |
| 12 | `12_experiments_geo_mask.ipynb` | Geographic inference mask |
| 13 | `13_birdclef_submission.ipynb` | Kaggle / Drive inference |

## Final model

| Setting | Value |
|---------|-------|
| Loss | Focal γ=1 |
| Weight decay | 1e-4 |
| Learning rate | 1e-3 |
| Context | 1 |
| TTA | 2.5 s stride |
| Epochs | 5 |
| CV AUC | 0.9771 ± 0.0005 |
| Kaggle | 0.798 public / 0.839 private |

Checkpoints and ONNX exports: `repro/outputs/models/best_model/`

## Kaggle submission

Upload `13_birdclef_submission.ipynb` to Kaggle and attach:

- Your MoE ONNX fold models as a dataset
- `perch_v2_no_dft.onnx` as a dataset (Kaggle has no internet at inference time)

On Colab/Drive, notebook 13 reads models from `repro/outputs/models/best_model/` and Perch from `BirdCLEF_Project/perch_v2_no_dft.onnx` (cached by notebook 02).

## Dependencies

```bash
pip install -r requirements.txt
```
