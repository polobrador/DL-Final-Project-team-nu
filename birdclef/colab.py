import os
import shutil
import subprocess
import sys
from pathlib import Path

from birdclef import paths
from birdclef.paths import configure, ensure_project_dirs

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


def set_kaggle_token(required: bool = False):
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    from google.colab import userdata

    try:
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    except userdata.SecretNotFoundError:
        if required:
            raise
        print("Kaggle token not set (optional for this notebook). Add KAGGLE_API_TOKEN in Colab secrets for notebook 02.")
    except userdata.NotebookAccessError:
        if required:
            raise
        print("Kaggle secret exists but notebook access is disabled. Enable it in Colab secrets for notebook 02.")


def ensure_perch_onnx() -> Path:
    if paths.PERCH_ONNX.exists():
        print(f"Using cached Perch ONNX at {paths.PERCH_ONNX}")
        return paths.PERCH_ONNX

    from huggingface_hub import hf_hub_download

    paths.PERCH_ONNX.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Perch v2 ONNX from Hugging Face ({paths.PERCH_HF_REPO}, ~394 MB)...")
    cached = Path(
        hf_hub_download(repo_id=paths.PERCH_HF_REPO, filename=paths.PERCH_HF_FILENAME)
    )
    shutil.copy2(cached, paths.PERCH_ONNX)
    print(f"Saved to {paths.PERCH_ONNX}")
    return paths.PERCH_ONNX


def _copy_csv(name: str, dst_dir: Path) -> None:
    dst = dst_dir / name
    if dst.exists():
        return
    for src in (paths.METADATA_DIR / name, paths.PROJECT_ROOT / name):
        if src.exists():
            shutil.copy(src, dst)
            print(f"Staged {name}")
            return
    raise FileNotFoundError(f"Missing {name}. Run notebook 02 first.")


def stage_metadata_csvs():
    for name in ("train.csv", "sample_submission.csv", "taxonomy.csv"):
        _copy_csv(name, paths.CONTENT)


def _unzip(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(target)], check=True)


def stage_baseline_embeddings():
    if any(paths.EMBEDDINGS_DIR.glob("*.npy")):
        return
    archive = paths.PROJECT_ROOT / "embeddings_v2_archive.zip"
    if archive.exists():
        local = paths.CONTENT / "embeddings_v2_archive.zip"
        shutil.copy(archive, local)
        _unzip(local, paths.EMBEDDINGS_DIR)
        print(f"Staged embeddings to {paths.EMBEDDINGS_DIR}")
        return
    raise FileNotFoundError(
        "Missing embeddings_v2_archive.zip. Run notebook 02 to extract embeddings first."
    )


def stage_tta_embeddings():
    if any(paths.EMBEDDINGS_TTA_DIR.glob("*.npy")):
        return
    archive = paths.PROJECT_ROOT / "embeddings_v2_TTA_archive.zip"
    if archive.exists():
        local = paths.CONTENT / "embeddings_v2_TTA_archive.zip"
        shutil.copy(archive, local)
        _unzip(local, paths.EMBEDDINGS_TTA_DIR)
        print(f"Staged TTA embeddings to {paths.EMBEDDINGS_TTA_DIR}")
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


def require_kaggle_token():
    set_kaggle_token(required=True)
