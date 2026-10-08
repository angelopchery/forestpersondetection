"""Video probing, normalization, tolerant frame reading and H.264 writing."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str
    rotation: int

    def to_dict(self) -> dict:
        return asdict(self)


def _ffprobe_rotation(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)]
    try:
        stream = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["streams"][0]
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        return 0
    if "rotate" in stream.get("tags", {}):
        return int(stream["tags"]["rotate"]) % 360
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(side["rotation"]) % 360
    return 0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 30.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = 0.0
        n_frames = stream.frames or int(round(duration * fps))
        return VideoInfo(
            path=str(path),
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            n_frames=n_frames,
            duration_s=duration,
            codec=stream.codec_context.name,
            rotation=_ffprobe_rotation(path),
        )


def needs_normalize(path: str | Path, max_fps: float | None = None) -> str | None:
    """Reason the clip must be re-encoded, or None if it can be read as is (saves a full re-encode)."""
    info = probe(path)
    if info.codec not in ("h264", "hevc"):
        return f"codec {info.codec}"
    if info.rotation:
        return f"rotation {info.rotation}"
    if max_fps and info.fps > max_fps + 0.5:
        return f"{info.fps:.1f} fps above max_fps"
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        avg, nominal = stream.average_rate, stream.guessed_rate or stream.base_rate
        if avg and nominal and abs(float(avg) - float(nominal)) > 0.01 * float(nominal):
            return "variable frame rate"
    return None


def normalize(src: str | Path, dst: str | Path, max_fps: float | None = None) -> Path:
    """Re-encode to constant-frame-rate H.264 with rotation applied, tolerating decode errors.

    Uploaded clips arrive with variable frame rate, rotation metadata, odd codecs and corrupt
    packets; normalizing once up front keeps every later stage simple.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    info = probe(src)
    fps = min(info.fps, max_fps) if max_fps else info.fps
    cmd = [
        ffmpeg, "-y", "-v", "error", "-err_detect", "ignore_err", "-i", str(src),
        "-vf", f"fps={fps:.6f},pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(dst)


def sample_frames(path: str | Path, n: int) -> list[np.ndarray]:
    """n frames spread evenly through the clip (skipping the first and last few percent)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0.05, 0.95, n) * max(total - 1, 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


class FrameReader:
    """Iterates BGR frames, skipping isolated corrupt frames instead of stopping early."""

    def __init__(self, path: str | Path, max_consecutive_failures: int = 30):
        self.path = Path(path)
        self.max_consecutive_failures = max_consecutive_failures
        self.skipped = 0

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {self.path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx, failures = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if total and idx < total - 1 and failures < self.max_consecutive_failures:
                        failures += 1
                        self.skipped += 1
                        idx += 1
                        continue
                    break
                failures = 0
                yield idx, frame
                idx += 1
        finally:
            cap.release()


class VideoWriter:
    """H.264 MP4 writer via PyAV (browser-playable, unlike OpenCV's mp4v)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.width, self.height = width - width % 2, height - height % 2
        self.container = av.open(str(path), mode="w")
        rate = Fraction(fps).limit_denominator(1001)
        for codec in ("libx264", "h264", "mpeg4"):
            try:
                self.stream = self.container.add_stream(codec, rate=rate)
                break
            except (av.FFmpegError, ValueError):
                continue
        else:
            raise RuntimeError("no usable video encoder found in PyAV")
        self.stream.width, self.stream.height = self.width, self.height
        self.stream.pix_fmt = "yuv420p"
        if self.stream.codec_context.name == "libx264":
            self.stream.options = {"crf": "20", "preset": "veryfast"}

    def write(self, bgr: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr[: self.height, : self.width]), format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
