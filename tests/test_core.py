"""Fast checks of the pure-logic parts (no GPU, no video): run with `python -m pytest tests`."""

import numpy as np

from grassdet.behavior import Track, _majority_smooth, posture_from_box, posture_from_keypoints
from grassdet.detect import merge_detections, slice_windows
from grassdet.evaluate import evaluate
from grassdet.postprocess import clean_tracks


def _kpts(shoulders, hips):
    k = np.zeros((17, 3))
    k[[5, 6, 11, 12], 2] = 1.0
    k[5, :2], k[6, :2] = shoulders
    k[11, :2], k[12, :2] = hips
    return k


def test_posture_from_keypoints():
    assert posture_from_keypoints(_kpts([(100, 50), (140, 50)], [(105, 150), (135, 150)])) == "upright"
    assert posture_from_keypoints(_kpts([(200, 100), (200, 120)], [(100, 105), (100, 125)])) == "horizontal"
    assert posture_from_keypoints(_kpts([(100, 100), (140, 100)], [(180, 160), (220, 160)])) == "crouched"  # ~53 deg
    hidden = _kpts([(100, 50), (140, 50)], [(105, 150), (135, 150)])
    hidden[[11, 12], 2] = 0.1  # hips hidden in grass
    assert posture_from_keypoints(hidden) is None


def test_posture_from_box():
    assert posture_from_box(np.array([0, 0, 200, 80])) == "horizontal"
    assert posture_from_box(np.array([0, 0, 50, 160])) == "upright"
    assert posture_from_box(np.array([0, 0, 100, 100])) == "unknown"


def test_majority_smooth_removes_flicker():
    labels = ["upright"] * 5 + ["horizontal"] + ["upright"] * 5
    assert _majority_smooth(labels, 5) == ["upright"] * 11


def test_clean_tracks_interpolates_and_filters():
    rows = [(f, 0, 0, 10, 20, 1, 0.9) for f in (0, 1, 2, 6, 7, 8)]  # gap of 3 frames
    rows += [(0, 50, 50, 60, 70, 2, 0.9)]                           # one-frame flicker
    tracks, stats = clean_tracks(rows, fps=10, min_track_seconds=0.5, min_mean_conf=0.25, max_interp_gap_seconds=1)
    assert stats["kept_tracks"] == 1 and stats["dropped_short"] == 1
    t = tracks[0]
    assert list(t.frames) == list(range(9)) and t.interpolated.sum() == 3
    assert isinstance(t, Track)


def test_slice_windows_cover_frame():
    wins = slice_windows(1920, 1080, 640, 0.2)
    covered = np.zeros((1080, 1920), bool)
    for x1, y1, x2, y2 in wins:
        covered[y1:y2, x1:x2] = True
    assert covered.all()


def test_merge_joins_tile_fragments():
    dets = np.array([[0, 0, 50, 100, 0.9, 0], [40, 0, 60, 100, 0.6, 0]], np.float32)  # fragment inside
    merged = merge_detections(dets, 0.5)
    assert len(merged) == 1 and merged[0, 2] == 60


def test_evaluate_perfect_and_missed():
    gt = {f: np.array([[1, 10, 10, 50, 90, 1.0]]) for f in range(10)}
    perfect = evaluate(gt, {f: g[:, :5] for f, g in gt.items()}, fps=10)
    assert perfect["idf1"] == 1.0 and perfect["recall"] == 1.0 and perfect["num_switches"] == 0
    half = evaluate(gt, {f: gt[f][:, :5] for f in range(5)}, fps=10)
    assert half["recall"] == 0.5 and half["recall_by_size"]["small (32-96px)"]["recall"] == 0.5
    switched = {f: np.array([[1 if f < 5 else 2, 10, 10, 50, 90]]) for f in range(10)}
    assert evaluate(gt, switched, fps=10)["num_switches"] == 1
