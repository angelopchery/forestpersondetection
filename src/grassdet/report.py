"""Clip-level quality checks that turn silent failures into explicit warnings."""

from __future__ import annotations

import cv2
import numpy as np

from .postprocess import Track

# Heuristic thresholds; revisit once labelled footage from the target cameras is available.
GRAYSCALE_CHANNEL_DIFF = 3.0   # mean |B-G|+|G-R| below this -> frames are effectively monochrome
LOW_LIGHT_BRIGHTNESS = 50.0    # mean gray level (0-255)
SMALL_PERSON_PX = 20.0         # median track box height in frame pixels
LOW_CONFIDENCE = 0.4           # median of per-track mean confidence


class FrameStats:
    """Samples cheap image statistics every N frames."""

    def __init__(self, every: int):
        self.every = max(1, every)
        self.channel_diff: list[float] = []
        self.brightness: list[float] = []

    def update(self, idx: int, frame: np.ndarray) -> None:
        if idx % self.every:
            return
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1]))).astype(np.int16)
        b, g, r = small[..., 0], small[..., 1], small[..., 2]
        self.channel_diff.append(float((np.abs(b - g) + np.abs(g - r)).mean()))
        self.brightness.append(float(cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2GRAY).mean()))

    def summary(self) -> dict:
        return {
            "mean_channel_diff": float(np.mean(self.channel_diff)) if self.channel_diff else None,
            "mean_brightness": float(np.mean(self.brightness)) if self.brightness else None,
        }


def quality_warnings(frame_stats: dict, tracks: list[Track], track_stats: dict, mode: str) -> list[str]:
    warnings = []
    diff, bright = frame_stats["mean_channel_diff"], frame_stats["mean_brightness"]
    if diff is not None and diff < GRAYSCALE_CHANNEL_DIFF:
        warnings.append(
            "Frames are effectively grayscale (thermal/IR/night-vision?). The detector is trained mostly on "
            "colour footage (thermal: drone views only); results may be unreliable - review manually."
        )
    if bright is not None and bright < LOW_LIGHT_BRIGHTNESS:
        warnings.append("Low-light footage: expect reduced recall.")
    if tracks:
        median_h = float(np.median([t.median_height for t in tracks]))
        if median_h < SMALL_PERSON_PX and mode != "sliced":
            warnings.append(
                f"People are very small (median box height {median_h:.0f}px). Re-run with inference.mode=sliced."
            )
        median_conf = float(np.median([t.mean_conf for t in tracks]))
        if median_conf < LOW_CONFIDENCE:
            warnings.append(
                f"Low model confidence (median track confidence {median_conf:.2f}): footage may differ from "
                "training conditions - review manually."
            )
    else:
        warnings.append("No people detected.")
        if track_stats.get("raw_tracks", 0) > 0:
            warnings.append(
                f"{track_stats['raw_tracks']} short-lived candidate tracks were discarded as flicker; "
                "if people are present, the footage is likely too hard for the current model."
            )
    return warnings
