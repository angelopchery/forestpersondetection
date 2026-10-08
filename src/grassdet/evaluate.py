"""Tracking evaluation against hand-labelled ground truth (MOTChallenge format, e.g. CVAT "MOT 1.1" export).

Reports the standard CLEAR-MOT / identity metrics plus the ones that matter for people in grass:
recall split by person size and by visibility, and false positives per minute of footage.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

SIZE_BUCKETS = {"tiny (<32px)": (0, 32), "small (32-96px)": (32, 96), "large (>=96px)": (96, np.inf)}
VIS_BUCKETS = {"heavily occluded (<0.3)": (0, 0.3), "partly occluded (0.3-0.7)": (0.3, 0.7), "visible (>=0.7)": (0.7, 1.01)}


def load_mot_gt(path: str | Path) -> dict[int, np.ndarray]:
    """gt.txt rows: frame(1-based), id, x, y, w, h, flag, class, visibility -> {frame0: (N, 6) id,x1,y1,x2,y2,vis}.

    Rows with flag 0 (MOT "ignore") are dropped. Missing visibility defaults to 1.
    """
    out: dict[int, list] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.replace(" ", "").split(",")]
        if len(v) > 6 and v[6] == 0:
            continue
        vis = v[8] if len(v) > 8 and v[8] >= 0 else 1.0
        out[int(v[0]) - 1].append((v[1], v[2], v[3], v[2] + v[4], v[3] + v[5], vis))
    return {f: np.array(r) for f, r in out.items()}


def load_predictions(path: str | Path, include_interpolated: bool = True) -> dict[int, np.ndarray]:
    """grassdet detections.csv (0-based frames) or MOT-format txt -> {frame0: (N, 5) id,x1,y1,x2,y2}."""
    path = Path(path)
    if path.suffix != ".csv":
        return {f: rows[:, :5] for f, rows in load_mot_gt(path).items()}
    out: dict[int, list] = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not include_interpolated and r.get("interpolated") == "1":
                continue
            out[int(r["frame"])].append(
                (float(r["track_id"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            )
    return {f: np.array(r) for f, r in out.items()}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N, 4) and (M, 4) x1y1x2y2 boxes."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _bucket_recall(hits: list[tuple[float, bool]], buckets: dict) -> dict:
    out = {}
    for name, (lo, hi) in buckets.items():
        sel = [hit for value, hit in hits if lo <= value < hi]
        out[name] = {"recall": round(float(np.mean(sel)), 4) if sel else None, "gt_boxes": len(sel)}
    return out


def evaluate(
    gt: dict[int, np.ndarray], pred: dict[int, np.ndarray], fps: float, iou_thresh: float = 0.5, name: str = "clip"
) -> dict:
    frames = sorted(set(gt) | set(pred))
    if not frames:
        raise ValueError("ground truth and predictions are both empty")
    n_frames = frames[-1] + 1
    acc = mm.MOTAccumulator(auto_id=False)
    size_hits, vis_hits = [], []
    empty_gt, empty_pred = np.zeros((0, 6)), np.zeros((0, 5))
    for f in range(n_frames):
        g, p = gt.get(f, empty_gt), pred.get(f, empty_pred)
        iou = _iou(g[:, 1:5], p[:, 1:5]) if len(g) and len(p) else np.zeros((len(g), len(p)))
        dist = np.where(iou >= iou_thresh, 1 - iou, np.nan)
        acc.update(g[:, 0].astype(int).tolist(), p[:, 0].astype(int).tolist(), dist, frameid=f)
        # Per-box hit/miss for the size and visibility breakdowns (one-to-one matching, IoU >= thresh).
        matched = np.zeros(len(g), dtype=bool)
        if len(g) and len(p):
            rows, cols = linear_sum_assignment(-iou)
            matched[rows[iou[rows, cols] >= iou_thresh]] = True
        for box, hit in zip(g, matched):
            size_hits.append((box[4] - box[2], hit))
            vis_hits.append((box[5], hit))

    mh = mm.metrics.create()
    names = ["mota", "motp", "idf1", "idp", "idr", "precision", "recall", "num_switches", "num_fragmentations",
             "mostly_tracked", "partially_tracked", "mostly_lost", "num_false_positives", "num_misses",
             "num_unique_objects"]
    summary = mh.compute(acc, metrics=names, name=name).iloc[0].to_dict()
    minutes = n_frames / fps / 60
    counts = {k for k in names if k.startswith(("num_", "mostly_", "partially_"))}
    result = {k: int(v) if k in counts else round(float(v), 4) for k, v in summary.items()}
    result["motp_iou"] = round(1 - result.pop("motp"), 4) if not np.isnan(summary["motp"]) else None
    result["false_positives_per_minute"] = round(result["num_false_positives"] / minutes, 2) if minutes else None
    result["recall_by_size"] = _bucket_recall(size_hits, SIZE_BUCKETS)
    if any(v < 1.0 for v, _ in vis_hits):  # only meaningful if the labels carry visibility
        result["recall_by_visibility"] = _bucket_recall(vis_hits, VIS_BUCKETS)
    result["frames"] = n_frames
    return result
