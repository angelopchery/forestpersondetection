"""Thin adapter around Ultralytics' trackers so they accept detections from any source
(full-frame or sliced inference), not only from model.track()."""

from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace


class Tracker:
    def __init__(self, config_path: str, fps: float):
        cfg = YAML.load(config_path)
        # Ultralytics counts track_buffer in frames; allow seconds so one config fits any frame rate.
        if "track_buffer_seconds" in cfg:
            cfg["track_buffer"] = max(1, int(round(cfg.pop("track_buffer_seconds") * fps)))
        cfg.setdefault("device", "cpu")  # only used by external ReID encoders
        self.gmc_max_side = cfg.pop("gmc_max_side", None)
        self._gmc_ready = False
        args = IterableSimpleNamespace(**cfg)
        if args.tracker_type not in TRACKER_MAP:
            raise ValueError(f"unknown tracker_type {args.tracker_type!r}; options: {sorted(TRACKER_MAP)}")
        self.tracker = TRACKER_MAP[args.tracker_type](args=args)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """dets: (N, 6) x1, y1, x2, y2, conf, cls -> (M, 6) x1, y1, x2, y2, track_id, conf of active tracks."""
        if not self._gmc_ready:
            gmc = getattr(self.tracker, "gmc", None)
            if gmc is not None and self.gmc_max_side:
                # Camera-motion estimation on full-resolution frames costs as much as detection;
                # a ~960px frame is plenty to estimate a global pan/shake.
                gmc.downscale = max(gmc.downscale, int(np.ceil(max(frame.shape[:2]) / self.gmc_max_side)))
            self._gmc_ready = True
        boxes = Boxes(np.asarray(dets, dtype=np.float32).reshape(-1, 6), frame.shape[:2])
        out = self.tracker.update(boxes, frame)
        if len(out) == 0:
            return np.zeros((0, 6), np.float32)
        return out[:, :6]
