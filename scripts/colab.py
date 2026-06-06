import os
import shutil
import subprocess
import sys
from pathlib import Path

from birdclef.paths import DRIVE_ROOT, EMBEDDINGS_DIR, REPRO_ROOT, ensure_project_dirs

CONTENT = Path("/content")


def mount_drive():
    from google.colab import drive

    if not DRIVE_ROOT.parent.exists():
        drive.mount("/content/drive")


def mount_and_configure() -> Path:
    mount_drive()
    if not (REPRO_ROOT / "birdclef").exists():
        raise FileNotFoundError(f"Clone repro into {REPRO_ROOT} on Google Drive.")
    root = str(REPRO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    ensure_project_dirs()
    return REPRO_ROOT


def set_kaggle_token():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    from google.colab import userdata

    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")


def _copy_csv(name: str, dst_dir: Path) -> None:
    dst = dst_dir / name
    if dst.exists():
        return
    for src in (DRIVE_ROOT / name, REPRO_ROOT / "data" / "metadata" / name):
        if src.exists():
            shutil.copy(src, dst)
            print(f"Staged {name}")
            return
    raise FileNotFoundError(f"Missing {name} on Drive or in repro/data/metadata/")


def stage_metadata_csvs():
    for name in ("train.csv", "sample_submission.csv", "taxonomy.csv"):
        _copy_csv(name, CONTENT)


def _unzip(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(target)], check=True)


def stage_baseline_embeddings():
    if any(EMBEDDINGS_DIR.glob("*.npy")):
        return
    archive = DRIVE_ROOT / "embeddings_v2_archive.zip"
    if archive.exists():
        local = CONTENT / "embeddings_v2_archive.zip"
        shutil.copy(archive, local)
        _unzip(local, EMBEDDINGS_DIR)
        print(f"Staged embeddings to {EMBEDDINGS_DIR}")
        return
    raise FileNotFoundError("Place embeddings_v2_archive.zip in BirdCLEF_Project on Drive.")


def stage_tta_embeddings():
    from birdclef.paths import EMBEDDINGS_TTA_DIR

    if any(EMBEDDINGS_TTA_DIR.glob("*.npy")):
        return
    archive = DRIVE_ROOT / "embeddings_v2_TTA_archive.zip"
    if archive.exists():
        local = CONTENT / "embeddings_v2_TTA_archive.zip"
        shutil.copy(archive, local)
        _unzip(local, EMBEDDINGS_TTA_DIR)
        print(f"Staged TTA embeddings to {EMBEDDINGS_TTA_DIR}")
        return
    raise FileNotFoundError("Place embeddings_v2_TTA_archive.zip in BirdCLEF_Project on Drive.")


def colab_prepare_training(stage_tta: bool = False) -> Path:
    mount_and_configure()
    stage_metadata_csvs()
    stage_baseline_embeddings()
    if stage_tta:
        stage_tta_embeddings()
    return REPRO_ROOT
