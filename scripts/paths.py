from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/BirdCLEF_Project")
REPRO_ROOT = DRIVE_ROOT / "repro"
CONTENT = Path("/content")

RAW_DATA_DIR = CONTENT / "bird_data"
METADATA_DIR = REPRO_ROOT / "data" / "metadata"
EMBEDDINGS_DIR = CONTENT / "embeddings_v2"
EMBEDDINGS_TTA_DIR = CONTENT / "embeddings_v2_TTA"
PERCH_ONNX = DRIVE_ROOT / "perch_v2_no_dft.onnx"
TRAIN_AUDIO_DIR = RAW_DATA_DIR / "train_audio"

TRAIN_CSV = CONTENT / "train.csv"
SAMPLE_SUBMISSION_CSV = CONTENT / "sample_submission.csv"
TAXONOMY_CSV = METADATA_DIR / "taxonomy.csv"

OUTPUT_ROOT = REPRO_ROOT / "outputs"
MODELS_DIR = OUTPUT_ROOT / "models"
FIGURES_DIR = OUTPUT_ROOT / "figures"
SWEEPS_DIR = OUTPUT_ROOT / "sweeps"
EXPERIMENTS_DIR = OUTPUT_ROOT / "experiments"
BEST_MODEL_DIR = MODELS_DIR / "best_model"

DRIVE_EMBEDDINGS_DIR = DRIVE_ROOT / "embeddings_v2"
DRIVE_EMBEDDINGS_TTA_DIR = DRIVE_ROOT / "embeddings_v2_TTA"


def ensure_project_dirs():
    for path in (
        METADATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        SWEEPS_DIR,
        EXPERIMENTS_DIR,
        BEST_MODEL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
