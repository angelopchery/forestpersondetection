"""Synthetic "people in grass" clip with exact MOT ground truth, for testing the evaluator and for A/B
comparisons of pipeline changes until real labelled footage exists.

People (segmented from Ultralytics' sample images) walk upright or crawl (rotated 90 degrees, slow)
across a procedural grass field, behind a foreground grass layer with dense clumps that fully hide
them for a while. The camera pans slowly. Scores on this clip are NOT a measure of real-world accuracy.

    python scripts/make_synthetic_clip.py --out data/samples/synthetic_grass.mp4
    -> data/samples/synthetic_grass.mp4 and data/samples/synthetic_grass_gt.txt (MOT format)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from grassdet.video_io import VideoWriter


def grass_field(h: int, w: int, rng: np.random.Generator, blades: int) -> np.ndarray:
    noise = cv2.resize(rng.random((h // 16, w // 16)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.stack([40 + 30 * noise, 90 + 50 * noise, 70 + 40 * noise], -1).astype(np.uint8)  # BGR earthy green
    for _ in range(blades):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        length, lean = int(rng.integers(10, 40)), int(rng.integers(-8, 9))
        g = int(rng.integers(80, 200))
        cv2.line(img, (x, y), (x + lean, y - length), (int(g * 0.35), g, int(g * 0.6)), 1, cv2.LINE_AA)
    return img


def foreground_grass(h: int, w: int, rng: np.random.Generator, clumps: list[tuple[int, int]]) -> tuple:
    """Sparse tall blades everywhere plus dense clumps at the given (x_center, width): (colour, alpha)."""
    color = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.float32)
    def blade(x, y, length, thick):
        g = int(rng.integers(60, 170))
        lean = int(rng.integers(-15, 16))
        cv2.line(color, (x, y), (x + lean, y - length), (int(g * 0.3), g, int(g * 0.5)), thick, cv2.LINE_AA)
        cv2.line(alpha, (x, y), (x + lean, y - length), 1.0, thick, cv2.LINE_AA)
    for _ in range(w // 3):
        blade(int(rng.integers(0, w)), int(rng.integers(h // 3, h)), int(rng.integers(40, 160)), 2)
    for cx, cw in clumps:
        for _ in range(cw * 6):
            blade(int(rng.integers(cx - cw // 2, cx + cw // 2)), h, int(rng.integers(h // 2, h)), 3)
    return color, alpha


def person_cutouts() -> list[tuple[np.ndarray, np.ndarray]]:
    seg = YOLO("yolo11n-seg.pt")
    cutouts = []
    for name in ("zidane.jpg", "bus.jpg"):
        img = cv2.imread(str(ASSETS / name))
        r = seg.predict(img, classes=[0], conf=0.5, retina_masks=True, verbose=False)[0]
        for box, mask in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.masks.data.cpu().numpy()):
            x1, y1, x2, y2 = box
            if y2 - y1 > 200:
                cutouts.append((img[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].astype(np.float32)))
    return cutouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/samples/synthetic_grass.mp4")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    H, W, WORLD_W = 1080, 1920, 2400
    n = int(args.seconds * args.fps)
    background = grass_field(H, WORLD_W, rng, blades=60000)
    fg_color, fg_alpha = foreground_grass(H, WORLD_W, rng, clumps=[(900, 220), (1700, 260)])
    cutouts = person_cutouts()

    # (cutout index, crawling?, body height px, start x, y of body bottom, speed px/s)
    actors = [
        (0, True, 110, 300, 820, 45.0),
        (1, False, 260, 1500, 760, -60.0),
        (2 % len(cutouts), True, 70, 1300, 560, 30.0),
        (3 % len(cutouts), False, 140, 600, 600, 35.0),
    ]
    sprites = []
    for idx, crawl, height, *_ in actors:
        img, mask = cutouts[idx]
        if crawl:  # lying on the side, head first in the direction of travel
            img, mask = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        s = height / img.shape[0]
        size = (max(1, int(img.shape[1] * s)), height)
        sprites.append((cv2.resize(img, size), cv2.resize(mask, size)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gt_lines = []
    with VideoWriter(out, args.fps, W, H) as writer:
        for f in range(n):
            t = f / args.fps
            cam_x = int((WORLD_W - W) * (0.5 - 0.5 * np.cos(np.pi * f / max(1, n - 1))))
            world = background.copy().astype(np.float32)
            person_masks = []
            for pid, ((spr, msk), (_, crawl, height, x0, y_bottom, speed)) in enumerate(zip(sprites, actors), 1):
                bob = 3 * np.sin(2 * np.pi * t * (1.0 if crawl else 2.0))
                x = int(x0 + speed * t)
                y = int(y_bottom - spr.shape[0] + bob)
                full = np.zeros((H, WORLD_W), np.float32)
                xs, ys = max(0, x), max(0, y)
                xe, ye = min(WORLD_W, x + spr.shape[1]), min(H, y + spr.shape[0])
                if xe <= xs or ye <= ys:
                    continue
                m = msk[ys - y : ye - y, xs - x : xe - x]
                world[ys:ye, xs:xe] = world[ys:ye, xs:xe] * (1 - m[..., None]) + spr[ys - y : ye - y, xs - x : xe - x] * m[..., None]
                full[ys:ye, xs:xe] = m
                person_masks.append((pid, full))
            a = fg_alpha[..., None]
            world = world * (1 - a) + fg_color * a
            frame = world[:, cam_x : cam_x + W].clip(0, 255).astype(np.uint8)
            writer.write(frame)

            for pid, full in person_masks:
                full_view = full[:, cam_x : cam_x + W] > 0.5
                visible = full_view & (fg_alpha[:, cam_x : cam_x + W] < 0.5)
                total = (full > 0.5).sum()
                if visible.sum() < 50 or total == 0:
                    continue  # fully hidden or out of view: no GT box (the evaluator then expects nothing)
                ys, xs = np.nonzero(visible)
                vis = visible.sum() / total
                # GT box = visible extent, as an annotator would draw it around a person in grass.
                gt_lines.append(f"{f + 1},{pid},{xs.min()},{ys.min()},{xs.max() - xs.min() + 1},"
                                f"{ys.max() - ys.min() + 1},1,1,{vis:.3f}")
    gt_path = out.with_name(out.stem + "_gt.txt")
    gt_path.write_text("\n".join(gt_lines) + "\n")
    print(f"wrote {n} frames -> {out}\nground truth -> {gt_path} ({len(gt_lines)} boxes)")


if __name__ == "__main__":
    main()
