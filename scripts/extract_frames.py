"""Sample frames from videos for annotation, skipping near-duplicates.

    python scripts/extract_frames.py --video data/raw_videos/ --out data/frames --fps 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from grassdet.video_io import FrameReader, probe

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".webm", ".wmv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="video file or folder")
    parser.add_argument("--out", default="data/frames")
    parser.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of video")
    parser.add_argument("--long-side", type=int, default=0, help="resize so the long side is this (0 = keep)")
    parser.add_argument("--min-diff", type=float, default=2.0,
                        help="skip a frame if its mean abs difference to the last saved frame is below this "
                             "(0-255 scale on a 160px thumbnail); 0 disables")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    src = Path(args.video)
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS) if src.is_dir() else [src]
    for video in videos:
        info = probe(video)
        step = max(1, int(round(info.fps / args.fps)))
        out_dir = Path(args.out) / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        last_thumb, saved, dupes = None, 0, 0
        for idx, frame in FrameReader(video):
            if idx % step:
                continue
            thumb = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if last_thumb is not None and args.min_diff > 0 and np.abs(thumb - last_thumb).mean() < args.min_diff:
                dupes += 1
                continue
            last_thumb = thumb
            if args.long_side:
                scale = args.long_side / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{video.stem}_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            saved += 1
        print(f"{video.name}: saved {saved} frames, skipped {dupes} near-duplicates -> {out_dir}")


if __name__ == "__main__":
    main()
