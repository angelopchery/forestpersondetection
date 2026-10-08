"""Download pretrained YOLO weights into models/.

    python scripts/download_models.py                 # yolo26s (stock detector), yolo11s-pose (behaviour stage)
    python scripts/download_models.py yolo26n.pt yolo11m.pt

The fine-tuned detector (grass_v1.pt) is not an Ultralytics asset: download it from the repository's
GitHub Releases into models/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

DEFAULT = ["yolo26s.pt", "yolo11s-pose.pt"]  # pose: behaviour stage


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    for name in sys.argv[1:] or DEFAULT:
        path = attempt_download_asset(models_dir / name)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
