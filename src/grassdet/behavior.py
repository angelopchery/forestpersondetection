"""Per-track posture and activity: crawling / prone_still / crouching / walking / standing / unknown.

Crawling is treated as a behaviour of a detected person, not a separate detector class, so the
detector stays general (people who stand up, kneel or lie still are still found) and this stage
works unchanged with any detector weights.

Posture comes from a pose model run on each track's crop (torso angle from shoulder/hip keypoints),
falling back to box shape when keypoints are unreliable (heavy occlusion, tiny people). Motion is
measured in body lengths per second after removing global camera translation, so a stationary person
in a panning shot is not reported as moving.

Known limits: top-down (drone) views have no meaningful "upright" direction, so most samples there
end up unknown; a person crawling straight toward the camera can look crouched.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .postprocess import Track
from .video_io import FrameReader

POSTURES = ("upright", "crouched", "horizontal", "unknown")
KPT_CONF = 0.4
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12  # COCO keypoint indices


def posture_from_keypoints(kpts: np.ndarray) -> str | None:
    """kpts: (17, 3) x, y, conf. None when the torso is not visible enough to judge."""
    def mid(i, j):
        pts = [kpts[k, :2] for k in (i, j) if kpts[k, 2] >= KPT_CONF]
        return np.mean(pts, axis=0) if pts else None

    shoulders, hips = mid(L_SH, R_SH), mid(L_HIP, R_HIP)
    if shoulders is None or hips is None:
        return None
    v = shoulders - hips
    length = float(np.hypot(*v))
    if length < 2:
        return None
    angle = np.degrees(np.arccos(np.clip(-v[1] / length, -1, 1)))  # 0 = shoulders straight above hips
    if angle >= 60:
        return "horizontal"
    if angle >= 35:
        return "crouched"
    if kpts[L_SH, 2] >= KPT_CONF and kpts[R_SH, 2] >= KPT_CONF:
        shoulder_w = float(np.hypot(*(kpts[L_SH, :2] - kpts[R_SH, :2])))
        if length < 0.8 * shoulder_w:  # torso foreshortened: bent over, or crawling toward the camera
            return "crouched"
    return "upright"


def posture_from_box(box: np.ndarray) -> str:
    w, h = box[2] - box[0], box[3] - box[1]
    aspect = w / max(h, 1e-6)
    if aspect > 1.3:
        return "horizontal"
    if aspect < 0.7:
        return "upright"
    return "unknown"  # square-ish boxes: crouching, partly hidden, or seen from above


def _majority_smooth(labels: list[str], window: int) -> list[str]:
    half = max(1, window) // 2
    out = []
    for i in range(len(labels)):
        votes = Counter(l for l in labels[max(0, i - half) : i + half + 1] if l != "unknown")
        out.append(votes.most_common(1)[0][0] if votes else "unknown")
    return out


class BehaviorAnalyzer:
    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.model = YOLO(cfg["pose_weights"])
        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        self.batch = cfg.get("pose_batch", 4)
        size = cfg.get("pose_imgsz", 320)
        self._pose_batch([np.zeros((size, size, 3), np.uint8)])  # warm-up: first call per shape is slow

    def _pose_batch(self, crops: list[np.ndarray]) -> list:
        """Runs in fixed-size batches (padded with blanks): every new batch shape costs seconds of CUDA
        kernel loading on first use, which dominated the runtime with naturally varying batch sizes."""
        size = self.cfg.get("pose_imgsz", 320)
        blank = np.zeros((size, size, 3), np.uint8)
        results = []
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            out = self.model.predict(chunk + [blank] * (self.batch - len(chunk)), imgsz=size, conf=0.25,
                                     device=self.device, quantize=16 if self.half else 32, verbose=False)
            results += out[: len(chunk)]
        return results

    def _square(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        """Pad to a square and resize to pose_imgsz: one fixed input shape keeps the GPU fast (varying
        shapes trigger re-planning on every call) and uniform scaling leaves torso angles unchanged."""
        size = self.cfg.get("pose_imgsz", 320)
        h, w = crop.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 114, np.uint8)
        canvas[:h, :w] = crop
        return cv2.resize(canvas, (size, size)), size / side

    def analyze(self, video, tracks: list[Track], fps: float) -> dict:
        """Adds .posture/.activity arrays to each track; returns clip-level stats."""
        every = max(1, int(round(fps / self.cfg.get("pose_samples_per_second", 5))))
        pad = self.cfg.get("crop_padding", 0.15)
        wanted: dict[int, list[tuple[Track, int]]] = defaultdict(list)
        for t in tracks:
            for k, (f, interp) in enumerate(zip(t.frames, t.interpolated)):
                if not interp and f % every == 0:
                    wanted[int(f)].append((t, k))

        samples: dict[int, dict[int, str]] = defaultdict(dict)  # track id -> {index in track: posture}
        cam = {}  # frame -> cumulative camera translation (x, y) in pixels
        prev_gray, offset = None, np.zeros(2)
        n_kpt = n_box = 0
        for idx, frame in FrameReader(video):
            small_w = 320
            scale = frame.shape[1] / small_w
            gray = cv2.cvtColor(cv2.resize(frame, (small_w, int(frame.shape[0] / scale))), cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            if prev_gray is not None:
                (dx, dy), response = cv2.phaseCorrelate(prev_gray, gray)
                if response > 0.1:  # unreliable estimates (e.g. blank frames) count as no motion
                    offset = offset + np.array([dx, dy]) * scale
            prev_gray, cam[idx] = gray, offset.copy()

            if idx not in wanted:
                continue
            crops, owners = [], []
            for t, k in wanted[idx]:
                x1, y1, x2, y2 = t.boxes[k]
                pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
                cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
                cx2, cy2 = int(min(frame.shape[1], x2 + pw)), int(min(frame.shape[0], y2 + ph))
                if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                    continue
                crop, crop_scale = self._square(frame[cy1:cy2, cx1:cx2])
                crops.append(crop)
                owners.append((t, k, np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]) * crop_scale))
            for (t, k, rel_box), result in zip(owners, self._pose_batch(crops) if crops else []):
                posture = None
                if result.keypoints is not None and len(result.boxes):
                    best = int(np.argmax(_box_overlap(result.boxes.xyxy.cpu().numpy(), rel_box)))
                    posture = posture_from_keypoints(result.keypoints.data[best].cpu().numpy())
                if posture is None:
                    posture, n_box = posture_from_box(t.boxes[k]), n_box + 1
                else:
                    n_kpt += 1
                samples[t.id][k] = posture

        window = max(1, int(round(fps * self.cfg.get("smooth_seconds", 1.0))))
        moving = self.cfg.get("moving_body_lengths_per_s", 0.1)
        for t in tracks:
            t.posture, t.activity = self._label_track(t, samples[t.id], cam, fps, window, moving)
        total = n_kpt + n_box
        return {"pose_samples": total, "keypoint_fraction": round(n_kpt / total, 3) if total else None}

    @staticmethod
    def _label_track(t: Track, samples: dict[int, str], cam: dict, fps: float, window: int, moving: float):
        n = len(t.frames)
        if samples:  # forward/backward fill sampled postures over every frame of the track
            keys = np.array(sorted(samples))
            nearest = keys[np.clip(np.searchsorted(keys, np.arange(n)), 0, len(keys) - 1)]
            posture = _majority_smooth([samples[int(i)] for i in nearest], window)
        else:
            posture = [posture_from_box(b) for b in t.boxes]

        # Speed in body lengths per second, measured in camera-compensated coordinates.
        centers = (t.boxes[:, :2] + t.boxes[:, 2:]) / 2
        offsets = np.array([cam.get(int(f), np.zeros(2)) for f in t.frames])
        world = centers - offsets  # phase correlation gives content motion; undo it to get scene coordinates
        body = max(float(np.median(np.max(t.boxes[:, 2:] - t.boxes[:, :2], axis=1))), 1.0)
        half = max(1, window // 2)
        activity = []
        for i in range(n):
            a, b = max(0, i - half), min(n - 1, i + half)
            dt = (t.frames[b] - t.frames[a]) / fps
            speed = np.hypot(*(world[b] - world[a])) / body / dt if dt > 0 else 0.0
            p, is_moving = posture[i], speed >= moving
            activity.append({
                "horizontal": "crawling" if is_moving else "prone_still",
                "crouched": "crouching",
                "upright": "walking" if is_moving else "standing",
            }.get(p, "unknown"))
        return np.array(posture), np.array(activity)


def _box_overlap(boxes: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(boxes[:, 2], ref[2]) - np.maximum(boxes[:, 0], ref[0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], ref[3]) - np.maximum(boxes[:, 1], ref[1]), 0, None)
    return ix * iy


def activity_summary(t: Track, fps: float) -> dict:
    if getattr(t, "activity", None) is None:
        return {}
    counts = Counter(t.activity.tolist())
    known = {k: v for k, v in counts.items() if k != "unknown"}
    return {
        "dominant_activity": max(known, key=known.get) if known else "unknown",
        "activity_seconds": {k: round(v / fps, 2) for k, v in sorted(counts.items())},
        "crawling_seconds": round(counts.get("crawling", 0) / fps, 2),
    }
