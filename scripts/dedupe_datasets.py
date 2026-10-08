"""Remove near-duplicate images across prepared datasets (data/datasets/*) before training.

Public person datasets are re-uploaded and re-mixed a lot (several Roboflow "SAR" sets are HERIDAL or
SARD copies, often with augmented duplicates). A copy of a validation image in the training set makes
validation scores meaningless, so duplicates are removed from train, never from val/test.

    python scripts/dedupe_datasets.py --datasets heridal sard rf_search_rescue      # report only
    python scripts/dedupe_datasets.py --datasets heridal sard rf_search_rescue --apply
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = kept when duplicates collide


def dhash(path: Path, size: int = 8) -> int | None:
    """64-bit difference hash: robust to resizing, re-compression and small colour changes."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, help="names under data/datasets/")
    parser.add_argument("--max-distance", type=int, default=4, help="Hamming distance counted as duplicate (0-64)")
    parser.add_argument("--cross-split-only", action="store_true",
                        help="only drop train images that duplicate a val/test image, keep near-duplicates within "
                             "train (static-camera sets like ForestPersons: the hash cannot see a small person "
                             "moving, so within-train 'duplicates' are real training signal)")
    parser.add_argument("--apply", action="store_true", help="delete duplicates (default: report only)")
    args = parser.parse_args()

    entries = []  # (hash, priority, dataset, split, image path)
    for name in args.datasets:
        root = Path("data/datasets") / name / "images"
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in split_dir.iterdir():
                h = dhash(img)
                if h is not None:
                    entries.append((h, SPLIT_PRIORITY.get(split_dir.name, 3), name, split_dir.name, img))
    entries.sort(key=lambda e: e[1])  # keep test/val copies, drop train copies
    hashes = np.array([e[0] for e in entries], dtype=np.uint64)

    # Bucket by each 16-bit quarter of the hash: two hashes within distance 4 share at least one quarter.
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        for q in range(4):
            buckets[(q, (int(h) >> (16 * q)) & 0xFFFF)].append(i)
    removed = set()
    pairs = defaultdict(int)
    for i in range(len(entries)):
        if i in removed or (args.cross_split_only and entries[i][3] == "train"):
            continue
        candidates = {j for q in range(4) for j in buckets[(q, (int(hashes[i]) >> (16 * q)) & 0xFFFF)] if j > i}
        for j in candidates:
            if j in removed or entries[j][3] != "train":
                continue
            if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= args.max_distance:
                removed.add(j)
                pairs[(entries[i][2], entries[i][3], entries[j][2], entries[j][3])] += 1

    print(f"{len(entries)} images scanned, {len(removed)} near-duplicates in train splits")
    for (d1, s1, d2, s2), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {d2}/{s2} duplicates {d1}/{s1}")
    if args.apply:
        for j in removed:
            img = entries[j][4]
            img.unlink(missing_ok=True)
            label = img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
            label.unlink(missing_ok=True)
        print("removed.")
    elif removed:
        print("report only: re-run with --apply to delete them")


if __name__ == "__main__":
    main()
