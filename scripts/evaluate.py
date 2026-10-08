"""Score one or more pipeline runs against hand-labelled ground truth, side by side.

    python scripts/evaluate.py --gt data/gt/clip01/gt.txt --pred outputs/clip01/detections.csv
    python scripts/evaluate.py --gt data/gt/clip01/gt.txt \
        --pred baseline=outputs/base/clip01/detections.csv tuned=outputs/tuned/clip01/detections.csv

Ground truth is MOTChallenge format (CVAT: Export -> "MOT 1.1"). Use the same comparison to accept
or reject any change (new weights, tracker settings, point-tracking gap filling, ...).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grassdet.evaluate import evaluate, load_mot_gt, load_predictions

HEADLINE = ["idf1", "mota", "recall", "precision", "num_switches", "num_fragmentations",
            "mostly_tracked", "mostly_lost", "false_positives_per_minute"]


def _fps_for(pred_path: Path, default: float | None) -> float:
    if default:
        return default
    tracks = pred_path.parent / "tracks.json"
    if tracks.exists():
        return float(json.loads(tracks.read_text())["fps"])
    raise SystemExit(f"--fps is required: no tracks.json next to {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="MOT gt.txt")
    parser.add_argument("--pred", nargs="+", required=True, metavar="[NAME=]PATH",
                        help="detections.csv from a pipeline run, or a MOT-format txt")
    parser.add_argument("--fps", type=float, help="clip frame rate (read from tracks.json if omitted)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a match")
    parser.add_argument("--no-interpolated", action="store_true", help="score observed boxes only")
    parser.add_argument("--out", help="write the full results as JSON here")
    args = parser.parse_args()

    gt = load_mot_gt(args.gt)
    results = {}
    for item in args.pred:
        name, _, path = item.rpartition("=")
        path = Path(path)
        name = name or path.parent.name
        pred = load_predictions(path, include_interpolated=not args.no_interpolated)
        results[name] = evaluate(gt, pred, _fps_for(path, args.fps), args.iou, name)

    width = max(10, max(len(n) for n in results) + 2)
    print(f"{'metric':<34}" + "".join(f"{n:>{width}}" for n in results))
    for key in HEADLINE:
        print(f"{key:<34}" + "".join(f"{str(r.get(key)):>{width}}" for r in results.values()))
    for group in ("recall_by_size", "recall_by_visibility"):
        first = next(iter(results.values()))
        for bucket in first.get(group, {}):
            label = f"recall {bucket}"
            print(f"{label:<34}" + "".join(f"{str(r[group][bucket]['recall']):>{width}}" for r in results.values()))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
