# BirdCLEF+ 2026

Reproducible training pipeline for the BirdCLEF+ competition. Runs on Google Colab with the project stored on Google Drive.

## Project setup

### 1. Download and unzip on Google Drive

Download the repository zip and unzip it anywhere on **My Drive**. The folder name does not matter (e.g. `birdclef-2026`, `repro-main`, etc.).

After unzipping you should have a single folder containing the notebooks, `birdclef/`, and `colab_init.py`:

```
your-project-folder/
├── colab_init.py
├── birdclef/
├── data/metadata/
├── outputs/
├── requirements.txt
├── README.md
└── *.ipynb
```

No extra parent folders are required.

### 2. Open a notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. **File → Open notebook → Google Drive** and open any notebook from the unzipped folder.
3. Select a **GPU** runtime (**Runtime → Change runtime type**).

Each notebook mounts Drive, locates the project folder automatically, and sets the working directory. You do not need to edit paths.

### 3. Kaggle API token (notebook 02 only)

Before running `02_download_and_extract_embeddings.ipynb`, add a Colab secret:

1. Click the key icon in the left sidebar.
2. Add a secret named `KAGGLE_API_TOKEN` with your [Kaggle API token](https://www.kaggle.com/settings).

### 4. Run the notebooks in order

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `01_data_exploration.ipynb` | Explore training metadata |
| 2 | `02_download_and_extract_embeddings.ipynb` | Download competition data, Perch ONNX, and extract train embeddings |
| 3–7 | `03`–`07` | Train baseline, ablation, and final models |
| 8–12 | `08`–`12` | Hyperparameter sweeps and experiments |
| 13 | `13_birdclef_submission.ipynb` | Generate Kaggle submission |

Notebook 02 must run before any training notebook. It downloads everything needed to get started; no manual file uploads are required.

## What gets created on Drive

After notebook 02, large artifacts are stored inside the project folder:

```
your-project-folder/
├── perch_v2_no_dft.onnx              # Perch v2 ONNX (~394 MB, cached)
├── embeddings_v2/                    # baseline embeddings (.npy)
├── embeddings_v2_TTA/                # TTA embeddings (.npy)
├── embeddings_v2_archive.zip         # optional archive (created at end of nb 02)
├── embeddings_v2_TTA_archive.zip     # optional archive
├── data/metadata/                    # train.csv, taxonomy.csv, etc.
└── outputs/                          # models, figures, sweep results
    └── models/best_model/            # final MoE checkpoints and ONNX exports
```

During training, embedding archives are copied to `/content/` on the Colab VM for faster I/O. Models and plots are saved under `outputs/` on Drive.

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

Checkpoints and ONNX exports: `outputs/models/best_model/`

## Kaggle submission

Upload `13_birdclef_submission.ipynb` to Kaggle and attach two datasets:

- Your 5-fold MoE ONNX models (`outputs/models/best_model/`)
- `perch_v2_no_dft.onnx` (cached on Drive by notebook 02)

Kaggle notebooks cannot access the internet at inference time, so Perch must be attached as a dataset.

For local testing on Colab/Drive, place test audio in `data/test_soundscapes/` and run notebook 13 from the project folder.

## Dependencies

Installed automatically in each notebook. Full list:

```bash
pip install -r requirements.txt
```
