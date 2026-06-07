"""Colab bootstrap: locate the project on Drive and add it to sys.path."""
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    cwd = Path(os.getcwd()).resolve()
    if (cwd / "birdclef").is_dir():
        return cwd

    my_drive = Path("/content/drive/MyDrive")
    if not my_drive.exists():
        raise FileNotFoundError("Mount Google Drive first.")

    matches = []
    for marker in my_drive.rglob("02_download_and_extract_embeddings.ipynb"):
        root = marker.parent.resolve()
        if (root / "birdclef").is_dir():
            matches.append(root)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return min(matches, key=lambda p: len(p.parts))

    raise FileNotFoundError(
        "Unzip the repository on Google Drive (My Drive), then open a notebook "
        "from that folder in Colab (File > Open notebook > Google Drive)."
    )


def bootstrap() -> Path:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    return root


PROJECT_ROOT = bootstrap()
