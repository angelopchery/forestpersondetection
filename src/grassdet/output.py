"""Result files: per-frame CSV, per-track JSON and the annotated video."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .behavior import activity_summary
from .postprocess import Track
from .video_io import FrameReader, VideoWriter


def write_detections_csv(tracks: list[Track], fps: float, path: Path) -> None:
    rows = []
    for t in tracks:
        n = len(t.frames)
        posture = t.posture if t.posture is not None else [""] * n
        activity = t.activity if t.activity is not None else [""] * n
        for f, box, conf, interp, p, a in zip(t.frames, t.boxes, t.confs, t.interpolated, posture, activity):
            rows.append((int(f), round(f / fps, 3), t.id, *np.round(box, 1).tolist(), round(float(conf), 4),
                         int(interp), p, a))
    rows.sort()
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "conf", "interpolated",
                         "posture", "activity"])
        writer.writerows(rows)


def track_summaries(tracks: list[Track], fps: float) -> list[dict]:
    return [
        {
            "track_id": t.id,
            "start_s": round(t.frames[0] / fps, 3),
            "end_s": round(t.frames[-1] / fps, 3),
            "duration_s": round((t.frames[-1] - t.frames[0] + 1) / fps, 3),
            "observed_frames": int((~t.interpolated).sum()),
            "interpolated_frames": int(t.interpolated.sum()),
            "mean_conf": round(t.mean_conf, 4),
            "median_box_height_px": round(t.median_height, 1),
            **activity_summary(t, fps),
        }
        for t in tracks
    ]


def write_tracks_json(tracks: list[Track], fps: float, path: Path) -> None:
    path.write_text(json.dumps({"fps": fps, "tracks": track_summaries(tracks, fps)}, indent=2))


def _color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def render_video(src: Path, tracks: list[Track], fps: float, dst: Path) -> None:
    """Second decoding pass that draws the cleaned tracks (interpolated boxes drawn thinner)."""
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    for t in tracks:
        activity = t.activity if t.activity is not None else [""] * len(t.frames)
        for f, box, conf, interp, act in zip(t.frames, t.boxes, t.confs, t.interpolated, activity):
            by_frame[int(f)].append((t.id, box, conf, interp, act))

    writer = None
    try:
        for idx, frame in FrameReader(src):
            if writer is None:
                writer = VideoWriter(dst, fps, frame.shape[1], frame.shape[0])
            scale = max(frame.shape[:2]) / 1080
            thick = max(1, int(round(2 * scale)))
            for track_id, box, conf, interp, act in by_frame.get(idx, []):
                x1, y1, x2, y2 = (int(v) for v in box)
                color = _color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if interp else thick)
                label = f"ID {track_id}" + ("" if interp else f" {conf:.2f}") + (f" {act}" if act else "")
                cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, thick)
            cv2.putText(frame, f"t={idx / fps:7.2f}s", (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * scale, (255, 255, 255), thick)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.close()
