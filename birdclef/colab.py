import os
import shutil
import subprocess
import sys
from pathlib import Path

from birdclef.paths import CONTENT, EMBEDDINGS_DIR, configure, ensure_project_dirs

PROJECT_ROOT: Path | None = None


def mount_drive():
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def mount_and_configure(project_root: Path | None = None) -> Path:
    global PROJECT_ROOT
    mount_drive()
    root = Path(project_root or os.getcwd()).resolve()
    if not (root / "birdclef").is_dir():
        raise FileNotFoundError(
            f"Expected birdclef/ inside {root}. Open the notebook from the project folder on Drive."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    configure(root)
    PROJECT_ROOT = root
    print(f"Project root: {root}")
    return root


def set_kaggle_token():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    from google.colab import userdata

    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")


def ensure_perch_onnx() -> Path:
    from birdclef.paths import PERCH_HF_FILENAME, PERCH_HF_REPO, PERCH_ONNX, PROJECT_ROOT

    if PERCH_ONNX.exists():
        print(f"Using cached Perch ONNX at {PERCH_ONNX}")
        return PERCH_ONNX

    from huggingface_hub import hf_hub_download

    PERCH_ONNX.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Perch v2 ONNX from Hugging Face ({PERCH_HF_REPO}, ~394 MB)...")
    cached = Path(
        hf_hub_download(repo_id=PERCH_HF_REPO, filename=PERCH_HF_FILENAME)
    )
    shutil.copy2(cached, PERCH_ONNX)
    print(f"Saved to {PERCH_ONNX}")
    return PERCH_ONNX


def _copy_csv(name: str, dst_dir: Path) -> None:
    from birdclef.paths import METADATA_DIR, PROJECT_ROOT

    dst = dst_dir / name
    if dst.exists():
        return
    for src in (METADATA_DIR / name, PROJECT_ROOT / name):
        if src.exists():
            shutil.copy(src, dst)
            print(f"Staged {name}")
            return
    raise FileNotFoundError(f"Missing {name}. Run notebook 02 first.")


def stage_metadata_csvs():
    for name in ("train.csv", "sample_submission.csv", "taxonomy.csv"):
        _copy_csv(name, CONTENT)


def _unzip(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(target)], check=True)


def stage_baseline_embeddings():
    from birdclef.paths import PROJECT_ROOT

    if any(EMBEDDINGS_DIR.glob("*.npy")):
        return
    archive = PROJECT_ROOT / "embeddings_v2_archive.zip"
    if archive.exists():
        local = CONTENT / "embeddings_v2_archive.zip"
        shutil.copy(archive, local)
        _unzip(local, EMBEDDINGS_DIR)
        print(f"Staged embeddings to {EMBEDDINGS_DIR}")
        return
    raise FileNotFoundError(
        "Missing embeddings_v2_archive.zip. Run notebook 02 to extract embeddings first."
    )


def stage_tta_embeddings():
    from birdclef.paths import EMBEDDINGS_TTA_DIR, PROJECT_ROOT

    if any(EMBEDDINGS_TTA_DIR.glob("*.npy")):
        return
    archive = PROJECT_ROOT / "embeddings_v2_TTA_archive.zip"
    if archive.exists():
        local = CONTENT / "embeddings_v2_TTA_archive.zip"
        shutil.copy(archive, local)
        _unzip(local, EMBEDDINGS_TTA_DIR)
        print(f"Staged TTA embeddings to {EMBEDDINGS_TTA_DIR}")
        return
    raise FileNotFoundError(
        "Missing embeddings_v2_TTA_archive.zip. Run notebook 02 to extract TTA embeddings first."
    )


def colab_prepare_training(stage_tta: bool = False) -> Path:
    root = mount_and_configure()
    stage_metadata_csvs()
    stage_baseline_embeddings()
    if stage_tta:
        stage_tta_embeddings()
    return root
