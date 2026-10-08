"""Create a short synthetic smoke-test clip (a slow pan over Ultralytics' bundled sample images).

It only proves the pipeline runs end to end; it says nothing about accuracy in grass.

    python scripts/make_test_clip.py --out data/samples/smoke_test.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from grassdet.video_io import VideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/samples/smoke_test.mp4")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    imgs = [cv2.imread(str(ASSETS / name)) for name in ("zidane.jpg", "bus.jpg")]
    h = 1080
    imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]
    canvas = np.concatenate(imgs, axis=1)  # wide strip to pan across
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = int(args.seconds * args.fps)
    width = 1920
    with VideoWriter(out, args.fps, width, h) as writer:
        for i in range(n):
            x = int((canvas.shape[1] - width) * i / max(1, n - 1))
            frame = canvas[:, x : x + width].copy()
            if i % 45 in range(10, 25):  # simulate a passing occluder (bar of "foliage")
                cv2.rectangle(frame, (700, 0), (900, h), (40, 110, 40), -1)
            writer.write(frame)
    print(f"wrote {n} frames -> {out}")


if __name__ == "__main__":
    main()
