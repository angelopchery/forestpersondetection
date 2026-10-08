"""Person detection: full-frame or sliced (tiled) inference with an Ultralytics YOLO model."""

from __future__ import annotations

import logging

import numpy as np
import torch
from ultralytics import YOLO

log = logging.getLogger(__name__)


def slice_windows(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping size x size windows covering the frame; edge windows are shifted inward."""
    step = max(1, int(size * (1 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= size:
            return [0]
        out = list(range(0, length - size, step))
        out.append(length - size)
        return out

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]


def merge_detections(dets: np.ndarray, ios_thresh: float) -> np.ndarray:
    """Greedy merge on intersection-over-smaller (as in SAHI's GREEDYNMM).

    Plain IoU-NMS cannot join the two partial boxes a tile border cuts a person into;
    IoS can, because the smaller fragment lies mostly inside the larger box.
    """
    if len(dets) < 2:
        return dets
    dets = dets[np.argsort(-dets[:, 4])]
    used = np.zeros(len(dets), dtype=bool)
    merged = []
    for i in range(len(dets)):
        if used[i]:
            continue
        box = dets[i].copy()
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            other = dets[j]
            iw = min(box[2], other[2]) - max(box[0], other[0])
            ih = min(box[3], other[3]) - max(box[1], other[1])
            if iw <= 0 or ih <= 0:
                continue
            smaller = min((box[2] - box[0]) * (box[3] - box[1]), (other[2] - other[0]) * (other[3] - other[1]))
            if smaller > 0 and iw * ih / smaller >= ios_thresh:
                box[:4] = [min(box[0], other[0]), min(box[1], other[1]), max(box[2], other[2]), max(box[3], other[3])]
                used[j] = True
        merged.append(box)
    return np.stack(merged)


class Detector:
    """Returns detections as an (N, 6) float array: x1, y1, x2, y2, conf, cls (frame pixels)."""

    def __init__(self, model_cfg: dict, inference_cfg: dict):
        self.model = YOLO(model_cfg["weights"])
        self.imgsz = model_cfg["imgsz"]
        self.conf = model_cfg["conf"]
        self.iou = model_cfg["iou"]
        self.classes = model_cfg.get("classes")
        self.device = model_cfg.get("device", 0) if torch.cuda.is_available() else "cpu"
        self.half = bool(model_cfg.get("half", True)) and self.device != "cpu"
        self.inf = inference_cfg
        self.slice_batch = inference_cfg["slice_batch"]
        self.warnings: list[str] = []

    def mode_for(self, width: int, height: int) -> str:
        mode = self.inf["mode"]
        if mode == "auto":
            return "sliced" if max(width, height) > self.inf["sliced_above_px"] else "full"
        return mode

    def probe_mode(self, frames: list[np.ndarray], conf: float = 0.3) -> tuple[str, dict]:
        """Pick full or sliced inference for a clip from a few sample frames.

        Resolution alone is a poor guide: a 1080p drone clip can hold 15px people that only tiling finds.
        Sliced wins when it finds clearly more confident people, or when the people found are small.
        """
        full_n = sliced_n = 0
        heights = []
        for frame in frames:
            full = self.predict(frame, "full")
            sliced = self.predict(frame, "sliced")
            full_n += int((full[:, 4] >= conf).sum())
            confident = sliced[sliced[:, 4] >= conf]
            sliced_n += len(confident)
            heights += (confident[:, 3] - confident[:, 1]).tolist()
        median_h = float(np.median(heights)) if heights else None
        small = median_h is not None and median_h < self.inf.get("small_person_px", 48)
        more = sliced_n >= max(full_n * 1.25, full_n + len(frames) * 0.5)
        mode = "sliced" if small or more else "full"
        return mode, {"probe_frames": len(frames), "full_dets": full_n, "sliced_dets": sliced_n,
                      "median_person_px": round(median_h, 1) if median_h else None, "chosen": mode}

    def _predict(self, images, imgsz: int) -> list:
        return self.model.predict(
            images, imgsz=imgsz, conf=self.conf, iou=self.iou, classes=self.classes,
            device=self.device, quantize=16 if self.half else 32, verbose=False,
        )

    @staticmethod
    def _to_array(result, dx: int = 0, dy: int = 0) -> np.ndarray:
        data = result.boxes.data.cpu().numpy()[:, :6].astype(np.float32)
        data[:, [0, 2]] += dx
        data[:, [1, 3]] += dy
        return data

    def _full(self, frame: np.ndarray) -> np.ndarray:
        return self._to_array(self._predict(frame, self.imgsz)[0])

    def _sliced(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        windows = slice_windows(w, h, self.inf["slice_size"], self.inf["slice_overlap"])
        parts = []
        for start in range(0, len(windows), self.slice_batch):
            batch = windows[start : start + self.slice_batch]
            crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in batch]
            for (x1, y1, _, _), result in zip(batch, self._predict(crops, self.inf["slice_size"])):
                parts.append(self._to_array(result, x1, y1))
        if self.inf.get("include_full_frame", True):
            parts.append(self._full(frame))
        dets = np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)
        return merge_detections(dets, self.inf["merge_ios"])

    def predict(self, frame: np.ndarray, mode: str) -> np.ndarray:
        """Detect people, degrading gracefully on GPU out-of-memory instead of failing the job."""
        while True:
            try:
                return self._sliced(frame) if mode == "sliced" else self._full(frame)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if mode == "sliced" and self.slice_batch > 1:
                    self.slice_batch = max(1, self.slice_batch // 2)
                    msg = f"GPU out of memory: slice batch reduced to {self.slice_batch}"
                elif self.device != "cpu":
                    self.device, self.half = "cpu", False
                    msg = "GPU out of memory: switched to CPU inference (much slower)"
                else:
                    raise
                log.warning(msg)
                self.warnings.append(msg)
