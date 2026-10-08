"""Measure detector speed and GPU memory on this machine (full-frame sizes and sliced 1080p/4K).

    python scripts/benchmark.py --weights models/yolo11s.pt
    python scripts/benchmark.py --weights models/yolo11s.engine --sizes 1280   # after TensorRT export
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from grassdet.detect import Detector


def bench(detector: Detector, frame: np.ndarray, mode: str, iters: int) -> tuple[float, float]:
    for _ in range(5):
        detector.predict(frame, mode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        detector.predict(frame, mode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fps = iters / (time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    return fps, mem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="models/yolo11s.pt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    frame_1080 = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame_4k = rng.integers(0, 255, (2160, 3840, 3), dtype=np.uint8)
    inference = {"mode": "full", "sliced_above_px": 2200, "slice_size": 640, "slice_overlap": 0.2,
                 "slice_batch": 8, "include_full_frame": True, "merge_ios": 0.5}
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {device} | weights: {args.weights} | half: {not args.no_half}")
    print(f"{'setting':<34}{'FPS':>8}{'peak MB':>10}")

    for size in args.sizes:
        model = {"weights": args.weights, "imgsz": size, "conf": 0.05, "iou": 0.6, "classes": [0],
                 "device": 0, "half": not args.no_half}
        fps, mem = bench(Detector(model, inference), frame_1080, "full", args.iters)
        print(f"{f'full 1080p @ imgsz {size}':<34}{fps:>8.1f}{mem:>10.0f}")

    model = {"weights": args.weights, "imgsz": 1280, "conf": 0.05, "iou": 0.6, "classes": [0],
             "device": 0, "half": not args.no_half}
    for name, frame in (("sliced 1080p (640 tiles)", frame_1080), ("sliced 4K (640 tiles)", frame_4k)):
        fps, mem = bench(Detector(model, inference), frame, "sliced", max(5, args.iters // 5))
        print(f"{name:<34}{fps:>8.1f}{mem:>10.0f}")
    print("Note: model-only throughput on random frames; end-to-end FPS (decode, tracking) is in each report.json.")


if __name__ == "__main__":
    main()
