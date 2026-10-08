"""End-to-end clip processing: normalize -> detect -> track -> clean -> outputs + QC report.

Every clip produces a report.json, including on failure, so no upload ends silently.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from .behavior import BehaviorAnalyzer
from .detect import Detector
from .output import render_video, track_summaries, write_detections_csv, write_tracks_json
from .postprocess import clean_tracks
from .report import FrameStats, quality_warnings
from .track import Tracker
from .video_io import FrameReader, needs_normalize, normalize, probe, sample_frames

log = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def process_clip(video: str | Path, cfg: dict, out_root: str | Path, detector: Detector | None = None) -> dict:
    video = Path(video)
    out_dir = Path(out_root) / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "input": str(video),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "warnings": [],
        "config": cfg,
    }
    t0 = time.perf_counter()
    stages: dict[str, float] = {}
    report["stage_seconds"] = stages
    normalized = out_dir / "_normalized.mp4"
    try:
        report["source"] = probe(video).to_dict()
        work_video = video
        mode_norm = cfg["video"].get("normalize", "auto")  # auto: only when the clip needs it
        reason = needs_normalize(video, cfg["video"].get("max_fps")) if mode_norm == "auto" else "forced"
        if mode_norm and reason:
            report["normalized_because"] = reason
            try:
                work_video = normalize(video, normalized, cfg["video"].get("max_fps"))
                stages["normalize"] = round(time.perf_counter() - t0, 2)
            except Exception as exc:  # fall back to reading the original file directly
                report["warnings"].append(f"Normalization failed ({exc!r}); reading original file directly.")
        info = probe(work_video)
        report["processed"] = info.to_dict()

        detector = detector or Detector(cfg["model"], cfg["inference"])
        mode = detector.mode_for(info.width, info.height)
        if cfg["inference"]["mode"] == "auto" and mode == "full" and cfg["inference"].get("probe_frames", 0):
            mode, report["mode_probe"] = detector.probe_mode(sample_frames(work_video, cfg["inference"]["probe_frames"]))
        report["inference_mode"] = mode
        tracker = Tracker(cfg["tracker"]["config"], info.fps)
        reader = FrameReader(work_video)
        frame_stats = FrameStats(cfg["output"].get("qc_sample_every", 15))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        rows, n_frames, n_dets = [], 0, 0
        t_infer = time.perf_counter()
        for idx, frame in tqdm(reader, total=info.n_frames or None, desc=video.name, unit="f"):
            dets = detector.predict(frame, mode)
            n_dets += len(dets)
            for x1, y1, x2, y2, track_id, conf in tracker.update(dets, frame):
                rows.append((idx, x1, y1, x2, y2, track_id, conf))
            frame_stats.update(idx, frame)
            n_frames += 1
        infer_s = time.perf_counter() - t_infer
        stages["detect_track"] = round(infer_s, 2)

        pp = cfg["postprocess"]
        tracks, track_stats = clean_tracks(
            rows, info.fps, pp["min_track_seconds"], pp["min_mean_conf"], pp["max_interp_gap_seconds"]
        )
        if cfg.get("behavior", {}).get("enabled", False) and tracks:
            analyzer = BehaviorAnalyzer(cfg["behavior"], cfg["model"].get("device", 0))
            t_stage = time.perf_counter()
            report["behavior"] = analyzer.analyze(work_video, tracks, info.fps)
            stages["behavior"] = round(time.perf_counter() - t_stage, 2)
        write_detections_csv(tracks, info.fps, out_dir / "detections.csv")
        write_tracks_json(tracks, info.fps, out_dir / "tracks.json")
        if cfg["output"].get("render_video", True):
            t_stage = time.perf_counter()
            render_video(work_video, tracks, info.fps, out_dir / "annotated.mp4")
            stages["render"] = round(time.perf_counter() - t_stage, 2)

        stats = frame_stats.summary()
        report["warnings"] += detector.warnings + quality_warnings(stats, tracks, track_stats, mode)
        if reader.skipped:
            report["warnings"].append(f"{reader.skipped} corrupt/undecodable frames were skipped.")
        report.update(
            status="ok",
            frames_processed=n_frames,
            frames_skipped=reader.skipped,
            raw_detections=n_dets,
            track_stats=track_stats,
            image_stats=stats,
            tracks=track_summaries(tracks, info.fps),
            inference_fps=round(n_frames / infer_s, 2) if infer_s > 0 else None,
            peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        )
    except Exception as exc:
        log.exception("processing failed for %s", video)
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        elapsed = time.perf_counter() - t0
        report["processing_time_s"] = round(elapsed, 2)
        duration = report.get("processed", report.get("source", {})).get("duration_s")
        report["real_time_factor"] = round(elapsed / duration, 3) if duration else None
        normalized.unlink(missing_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
