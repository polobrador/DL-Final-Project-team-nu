"""Colab bootstrap: add the project to sys.path and set the working directory."""
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    cwd = Path(os.getcwd()).resolve()
    if (cwd / "birdclef").is_dir() and (cwd / "colab_init.py").exists():
        return cwd

    my_drive = Path("/content/drive/MyDrive")
    if my_drive.exists():
        matches = []
        for init_file in my_drive.rglob("colab_init.py"):
            root = init_file.parent.resolve()
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


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
