from pathlib import Path

CONTENT = Path("/content")

PERCH_HF_REPO = "justinchuby/Perch-onnx"
PERCH_HF_FILENAME = "perch_v2_no_dft.onnx"

PROJECT_ROOT: Path | None = None
RAW_DATA_DIR: Path | None = None
METADATA_DIR: Path | None = None
EMBEDDINGS_DIR: Path | None = None
EMBEDDINGS_TTA_DIR: Path | None = None
PERCH_ONNX: Path | None = None
TRAIN_AUDIO_DIR: Path | None = None
TRAIN_CSV: Path | None = None
SAMPLE_SUBMISSION_CSV: Path | None = None
TAXONOMY_CSV: Path | None = None
OUTPUT_ROOT: Path | None = None
MODELS_DIR: Path | None = None
FIGURES_DIR: Path | None = None
SWEEPS_DIR: Path | None = None
EXPERIMENTS_DIR: Path | None = None
BEST_MODEL_DIR: Path | None = None
EMBEDDINGS_ARCHIVE_DIR: Path | None = None
EMBEDDINGS_TTA_ARCHIVE_DIR: Path | None = None
TEST_SOUNDSCAPES_DIR: Path | None = None


def configure(project_root: Path) -> Path:
    global PROJECT_ROOT, RAW_DATA_DIR, METADATA_DIR, EMBEDDINGS_DIR
    global EMBEDDINGS_TTA_DIR, PERCH_ONNX, TRAIN_AUDIO_DIR, TRAIN_CSV
    global SAMPLE_SUBMISSION_CSV, TAXONOMY_CSV, OUTPUT_ROOT, MODELS_DIR
    global FIGURES_DIR, SWEEPS_DIR, EXPERIMENTS_DIR, BEST_MODEL_DIR
    global EMBEDDINGS_ARCHIVE_DIR, EMBEDDINGS_TTA_ARCHIVE_DIR, TEST_SOUNDSCAPES_DIR

    root = Path(project_root).resolve()
    PROJECT_ROOT = root
    RAW_DATA_DIR = CONTENT / "bird_data"
    METADATA_DIR = root / "data" / "metadata"
    EMBEDDINGS_DIR = CONTENT / "embeddings_v2"
    EMBEDDINGS_TTA_DIR = CONTENT / "embeddings_v2_TTA"
    PERCH_ONNX = root / PERCH_HF_FILENAME
    TRAIN_AUDIO_DIR = RAW_DATA_DIR / "train_audio"
    TRAIN_CSV = CONTENT / "train.csv"
    SAMPLE_SUBMISSION_CSV = CONTENT / "sample_submission.csv"
    TAXONOMY_CSV = METADATA_DIR / "taxonomy.csv"
    OUTPUT_ROOT = root / "outputs"
    MODELS_DIR = OUTPUT_ROOT / "models"
    FIGURES_DIR = OUTPUT_ROOT / "figures"
    SWEEPS_DIR = OUTPUT_ROOT / "sweeps"
    EXPERIMENTS_DIR = OUTPUT_ROOT / "experiments"
    BEST_MODEL_DIR = MODELS_DIR / "best_model"
    EMBEDDINGS_ARCHIVE_DIR = root / "embeddings_v2"
    EMBEDDINGS_TTA_ARCHIVE_DIR = root / "embeddings_v2_TTA"
    TEST_SOUNDSCAPES_DIR = root / "data" / "test_soundscapes"
    ensure_project_dirs()
    return root


def ensure_project_dirs():
    if PROJECT_ROOT is None:
        return
    for path in (
        METADATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        SWEEPS_DIR,
        EXPERIMENTS_DIR,
        BEST_MODEL_DIR,
        EMBEDDINGS_ARCHIVE_DIR,
        EMBEDDINGS_TTA_ARCHIVE_DIR,
        TEST_SOUNDSCAPES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
