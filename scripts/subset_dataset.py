"""Make a lighter copy of a prepared dataset (data/datasets/<name>) by keeping a fraction of its train images.

Val/test splits are kept whole so scores stay comparable. Images are hard-linked (same drive, no extra
space) and labels copied, so the subset costs almost no disk. Selection is deterministic (hash of the
file stem), so re-running gives the same subset.

    python scripts/subset_dataset.py visdrone visdrone_lite --fraction 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import yaml


def keep(stem: str, fraction: float) -> bool:
    return int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:  # different drive or filesystem without hard links
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="name under data/datasets/")
    parser.add_argument("name", help="name of the subset to create")
    parser.add_argument("--fraction", type=float, required=True, help="share of train images to keep (0-1)")
    args = parser.parse_args()

    src = Path("data/datasets") / args.source
    out = Path("data/datasets") / args.name
    meta = yaml.safe_load((src / "dataset.yaml").read_text())
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for split in ("train", "val", "test"):
        if split not in meta:
            continue
        n = {"images": 0, "negatives": 0, "boxes": 0}
        for img in sorted((src / "images" / split).iterdir()):
            if split == "train" and not keep(img.stem, args.fraction):
                continue
            label = src / "labels" / split / f"{img.stem}.txt"
            link_or_copy(img, out / "images" / split / img.name)
            lines = label.read_text().split() if label.exists() else []
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, out / "labels" / split / label.name) if label.exists() else None
            n["images"] += 1
            n["boxes"] += len(lines) // 5
            n["negatives"] += not lines
        counts[split] = n
    meta.update(path=str(out.resolve()), counts=counts, subset_of=args.source, fraction=args.fraction)
    (out / "dataset.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if (src / "sources.csv").exists():
        shutil.copy2(src / "sources.csv", out / "sources.csv")
    for split, n in counts.items():
        print(f"{split}: {n['images']} images, {n['boxes']} boxes, {n['negatives']} negatives")


if __name__ == "__main__":
    main()
