"""Offline track clean-up: drop flickering tracks and interpolate through occlusion gaps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Track:
    id: int
    source_id: int            # tracker-assigned ID before renumbering
    frames: np.ndarray        # (N,) int
    boxes: np.ndarray         # (N, 4) x1, y1, x2, y2
    confs: np.ndarray         # (N,) detection confidence; 0 for interpolated points
    interpolated: np.ndarray  # (N,) bool
    posture: Optional[np.ndarray] = None   # (N,) str, filled by behavior.BehaviorAnalyzer
    activity: Optional[np.ndarray] = None  # (N,) str

    @property
    def mean_conf(self) -> float:
        observed = self.confs[~self.interpolated]
        return float(observed.mean()) if len(observed) else 0.0

    @property
    def median_height(self) -> float:
        return float(np.median(self.boxes[:, 3] - self.boxes[:, 1]))


def group_rows(rows: list[tuple]) -> dict[int, list[tuple]]:
    """rows: (frame, x1, y1, x2, y2, track_id, conf) -> {track_id: rows sorted by frame}."""
    tracks: dict[int, list[tuple]] = defaultdict(list)
    for row in rows:
        tracks[int(row[5])].append(row)
    return {tid: sorted(r, key=lambda x: x[0]) for tid, r in tracks.items()}


def _interpolate(frames: np.ndarray, boxes: np.ndarray, confs: np.ndarray, max_gap: int) -> tuple[np.ndarray, ...]:
    """Linearly fill gaps of up to max_gap missing frames; filled points are flagged."""
    out_f, out_b, out_c, out_i = [frames[0]], [boxes[0]], [confs[0]], [False]
    for k in range(1, len(frames)):
        gap = frames[k] - frames[k - 1]
        if 1 < gap <= max_gap + 1:
            for f in range(frames[k - 1] + 1, frames[k]):
                t = (f - frames[k - 1]) / gap
                out_f.append(f)
                out_b.append((1 - t) * boxes[k - 1] + t * boxes[k])
                out_c.append(0.0)
                out_i.append(True)
        out_f.append(frames[k])
        out_b.append(boxes[k])
        out_c.append(confs[k])
        out_i.append(False)
    return np.array(out_f), np.array(out_b), np.array(out_c), np.array(out_i)


def clean_tracks(
    rows: list[tuple],
    fps: float,
    min_track_seconds: float,
    min_mean_conf: float,
    max_interp_gap_seconds: float,
) -> tuple[list[Track], dict]:
    """Filter short / low-confidence tracks, fill short gaps, renumber IDs 1..K by first appearance."""
    min_frames = max(1, int(round(min_track_seconds * fps)))
    max_gap = int(round(max_interp_gap_seconds * fps))
    kept: list[Track] = []
    dropped_short = dropped_conf = 0
    for source_id, track_rows in group_rows(rows).items():
        arr = np.array(track_rows, dtype=np.float64)
        frames = arr[:, 0].astype(int)
        boxes, confs = arr[:, 1:5], arr[:, 6]
        if frames[-1] - frames[0] + 1 < min_frames:
            dropped_short += 1
            continue
        if confs.mean() < min_mean_conf:
            dropped_conf += 1
            continue
        f, b, c, i = _interpolate(frames, boxes, confs, max_gap)
        kept.append(Track(id=0, source_id=source_id, frames=f, boxes=b, confs=c, interpolated=i))
    kept.sort(key=lambda t: t.frames[0])
    for new_id, track in enumerate(kept, start=1):
        track.id = new_id
    stats = {
        "raw_tracks": len(set(int(r[5]) for r in rows)),
        "kept_tracks": len(kept),
        "dropped_short": dropped_short,
        "dropped_low_conf": dropped_conf,
    }
    return kept, stats
