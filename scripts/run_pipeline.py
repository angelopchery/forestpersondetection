"""Process one video or every video in a folder.

    python scripts/run_pipeline.py --video data/samples/clip.mp4
    python scripts/run_pipeline.py --video data/raw_videos/ --set inference.mode=sliced
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from grassdet.detect import Detector
from grassdet.pipeline import load_config, process_clip

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".webm", ".wmv"}


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """--set section.key=value (value parsed as YAML, so numbers/bools/lists work)."""
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="video file or folder of videos")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    src = Path(args.video)
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS) if src.is_dir() else [src]
    if not videos:
        raise SystemExit(f"no videos found in {src}")

    detector = Detector(cfg["model"], cfg["inference"])  # load the model once for all clips
    for video in videos:
        report = process_clip(video, cfg, args.out, detector)
        print(f"\n[{report['status'].upper()}] {video.name}: "
              f"{report.get('track_stats', {}).get('kept_tracks', 0)} tracks, "
              f"{report.get('inference_fps')} fps, RTF {report.get('real_time_factor')}")
        for w in report["warnings"]:
            print(f"  ! {w}")
        if report["status"] == "failed":
            print(f"  error: {report['error']}")
        print(f"  -> {Path(args.out) / video.stem}")


if __name__ == "__main__":
    main()
