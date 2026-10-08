"""Convert a dataset (YOLO, COCO, Pascal VOC, or labelled videos in MOT format) into a single-class
"person" YOLO dataset under data/datasets/<name>/, with its licence recorded next to it.

Splits are made per scene/video (never per frame): neighbouring frames are near-duplicates, and a
per-frame split leaks them into validation and inflates scores. --keep-splits uses the source's own
train/val/test folders or annotation files instead (official test sets stay test sets).

    python scripts/prepare_dataset.py --format mot --src data/labelled_videos --name own_clips --licence "own"
    python scripts/prepare_dataset.py --format coco --src data/external/forestpersons --ann-glob "**/*.json" \
        --name forestpersons --licence "CC BY-NC-SA 4.0" --noncommercial
    python scripts/prepare_dataset.py --format yolo --src data/external/hit_uav --names-from-yaml data.yaml \
        --keep-splits --name hit_uav --licence "CC BY 4.0"
    python scripts/prepare_dataset.py --format voc --src D:/HERIDAL --name heridal --licence "CC BY 3.0" --tile 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts"}
PERSON_NAMES = {
    "person", "persons", "people", "pedestrian", "human", "humans", "man", "woman", "civilian", "soldier",
    "soldiers", "camouflaged_soldier", "crawling_person", "lying_person", "partially_visible_person",
    "laying down", "laying_down", "running", "walking", "standing", "sitting", "lying",
}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "test": "test", "testing": "test"}
ROBOFLOW_SUFFIX = re.compile(r"(_(jpg|jpeg|png|bmp))?\.rf\.[0-9A-Za-z]+$")  # frame_0001_jpg.rf.<hash>


@dataclass
class Item:
    source: object                    # image Path, or (video Path, frame index)
    boxes: list = field(default_factory=list)  # (x1, y1, x2, y2); pixels, or 0-1 if normalized
    group: str = ""                   # scene/sequence id used for splitting
    split: str | None = None          # split given by the source, if any
    normalized: bool = False


def is_person(name: str) -> bool:
    return name.strip().lower().replace("-", "_") in {n.replace("-", "_") for n in PERSON_NAMES}


def split_hint(parts) -> str | None:
    for part in reversed([str(p).lower() for p in parts]):
        if part in SPLIT_ALIASES:
            return SPLIT_ALIASES[part]
    return None


def make_group(path: Path, root: Path, args) -> str:
    """Scene id: the first --group-depth folders, or folder + stem minus Roboflow suffix and frame number."""
    rel = path.relative_to(root)
    folders = [p for p in rel.parent.parts if p.lower() not in SPLIT_ALIASES and p.lower() not in ("images", "labels")]
    if args.group_depth:
        return "/".join(folders[: args.group_depth]) or rel.stem
    stem = ROBOFLOW_SUFFIX.sub("", rel.stem)
    stem = re.sub(args.group_regex, "", stem) or stem
    return "/".join(folders + [stem])


def from_yolo(src: Path, args):
    """YOLO images/labels trees. Class names come from --names or from each data.yaml found (--names-from-yaml),
    so multi-part exports with different class lists are handled."""
    yaml_names: dict[Path, list[str]] = {}
    if args.names_from_yaml:
        for y in src.rglob(args.names_from_yaml):
            names = yaml.safe_load(y.read_text()).get("names", [])
            yaml_names[y.parent] = list(names.values()) if isinstance(names, dict) else list(names)
    fixed = (args.names or "person").split(",")
    seen_originals = set()
    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS):
        parts = list(img.parts)
        if "images" not in parts or ".cache" in parts:
            continue
        if ROBOFLOW_SUFFIX.search(img.stem):
            # Roboflow exports often hold several augmented copies of one frame (x.jpg.rf.<hash1>, ...<hash2>).
            # Keep one: training applies its own augmentation, and the copies only inflate the dataset.
            original = (img.parent, ROBOFLOW_SUFFIX.sub("", img.stem))
            if original in seen_originals:
                continue
            seen_originals.add(original)
        idx = len(parts) - 1 - parts[::-1].index("images")
        label = Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
        names = fixed
        for parent in img.parents:
            if parent in yaml_names:
                names = yaml_names[parent]
                break
        person_ids = {i for i, n in enumerate(names) if is_person(n)}
        boxes = []
        if label.exists():
            for line in label.read_text().splitlines():
                v = line.split()
                if len(v) == 5 and int(v[0]) in person_ids:
                    cx, cy, bw, bh = (float(x) for x in v[1:5])
                    boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
                elif len(v) > 5 and int(v[0]) in person_ids:  # polygon (segmentation export) -> its box
                    xs, ys = [float(x) for x in v[1::2]], [float(y) for y in v[2::2]]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
        yield Item(img, boxes, make_group(img, src, args), split_hint(img.relative_to(src).parts[:-1]), True)


def from_coco(src: Path, args):
    anns = [Path(args.ann)] if args.ann else sorted(src.glob(args.ann_glob))
    if not anns:
        raise SystemExit(f"no COCO annotation files found under {src} ({args.ann_glob})")
    index = {}  # basename -> path, for annotation files whose file_name paths don't match the folder layout
    for ann in anns:
        data = json.loads(ann.read_text())
        if "images" not in data or "annotations" not in data:
            continue
        hint = split_hint([ann.stem.lower()] + re.split(r"[_\-.]", ann.stem.lower()))
        # ForestPersons names its classes pose_visibility (sitting_100, lying_40, ...) under supercategory "person";
        # "unknown_0" is a fully hidden person (0 % visible): not learnable as a box, and an image holding one is not
        # a true negative either, so those images are skipped.
        cats = data.get("categories", [])
        hidden_cats = {c["id"] for c in cats if c["name"].lower().startswith("unknown")}
        person_cats = {c["id"] for c in cats
                       if (is_person(c["name"]) or is_person(str(c.get("supercategory", "")))) and c["id"] not in hidden_cats}
        if not person_cats:
            raise SystemExit(f"{ann.name}: no person categories among {[c['name'] for c in cats]}")
        boxes = defaultdict(list)
        hidden_imgs = set()
        for a in data["annotations"]:
            if a["category_id"] in hidden_cats:
                hidden_imgs.add(a["image_id"])
            elif a["category_id"] in person_cats and not a.get("ignore", 0) and not a.get("iscrowd", 0):
                x, y, w, h = a["bbox"]
                boxes[a["image_id"]].append((x, y, x + w, y + h))
        missing = 0
        for im in data["images"]:
            if im["id"] in hidden_imgs:
                continue
            path = src / im["file_name"]
            if not path.exists():
                if not index:
                    index = {p.name: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS}
                path = index.get(Path(im["file_name"]).name)
                if path is None:
                    missing += 1  # not downloaded (subset) or absent
                    continue
            yield Item(path, boxes[im["id"]], make_group(path, src, args), hint)
        print(f"  {ann.name}: {len(data['images']) - missing} images found, {missing} not present locally")


def from_voc(src: Path, args):
    for xml in sorted(src.rglob("*.xml")):
        root = ET.parse(xml).getroot()
        fname = root.findtext("filename") or ""
        candidates = [xml.with_name(fname)] + [xml.with_suffix(e) for e in IMG_EXTS]
        candidates += [xml.parent.parent / d / fname for d in ("JPEGImages", "images")]
        img = next((c for c in candidates if c.is_file()), None)
        if img is None:
            continue
        boxes = []
        for obj in root.iter("object"):
            if is_person(obj.findtext("name") or ""):
                bb = obj.find("bndbox")
                boxes.append(tuple(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))
        yield Item(img, boxes, make_group(img, src, args), split_hint(img.relative_to(src).parts[:-1]))


def from_mot(src: Path, args):
    """Videos with MOT ground truth: <clip>.mp4 + (<clip>/gt/gt.txt | <clip>_gt.txt | <clip>.txt)."""
    for video in sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
        gt = next((g for g in (video.parent / video.stem / "gt" / "gt.txt", video.with_name(video.stem + "_gt.txt"),
                               video.with_suffix(".txt")) if g.exists()), None)
        if gt is None:
            print(f"  skip {video.name}: no ground truth found")
            continue
        boxes = defaultdict(list)
        for line in gt.read_text().splitlines():
            v = [float(x) for x in line.split(",")] if line.strip() else []
            if len(v) >= 6 and (len(v) < 7 or v[6] != 0):
                boxes[int(v[0]) - 1].append((v[2], v[3], v[2] + v[4], v[3] + v[5]))
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for f in range(0, n, args.every):
            yield Item((video, f), boxes.get(f, []), str(video.relative_to(src).with_suffix("")))


def split_of(group: str, val: float, test: float) -> str:
    r = _unit_hash(group)
    return "test" if r < test else "val" if r < test + val else "train"


def _unit_hash(text: str) -> float:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def pieces(img, boxes, args):
    """The whole image (optionally downscaled), or overlapping tiles for huge drone/SAR images.

    Tiling keeps small people at native resolution: a 20px person in a 4000px image would shrink
    to ~3px if the whole image were resized to 640 for training.
    """
    h, w = img.shape[:2]
    if args.tile and max(h, w) > args.tile * 1.5:
        step = int(args.tile * (1 - args.tile_overlap))
        xs = list(range(0, max(1, w - args.tile), step)) + [max(0, w - args.tile)]
        ys = list(range(0, max(1, h - args.tile), step)) + [max(0, h - args.tile)]
        for ty in sorted(set(ys)):
            for tx in sorted(set(xs)):
                tw, th = min(args.tile, w - tx), min(args.tile, h - ty)
                kept = []
                for x1, y1, x2, y2 in boxes:
                    cx1, cy1, cx2, cy2 = max(x1, tx), max(y1, ty), min(x2, tx + tw), min(y2, ty + th)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    # keep a box cut by the tile border only if most of the person is inside the tile
                    if (cx2 - cx1) * (cy2 - cy1) >= 0.6 * (x2 - x1) * (y2 - y1):
                        kept.append((cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty))
                yield f"_{tx}_{ty}", img[ty : ty + th, tx : tx + tw], kept
        return
    scale = min(1.0, args.max_side / max(h, w)) if args.max_side else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        boxes = [tuple(v * scale for v in b) for b in boxes]
    yield "", img, boxes


def write_piece(out, split, stem, img, boxes, args, state, counts) -> bool:
    h, w = img.shape[:2]
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 >= args.min_box_px and y2 - y1 >= args.min_box_px:
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    if not lines:
        # keep person-free images (hard negatives) only up to the cap, relative to positives so far
        if state["neg"] >= args.max_negative_frac * max(state["pos"], 1) / (1 - args.max_negative_frac):
            return False
        state["neg"] += 1
    else:
        state["pos"] += 1
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    c = counts[split]
    c["images"] += 1
    c["boxes"] += len(lines)
    c["negatives"] += not lines
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", required=True, choices=["yolo", "coco", "voc", "mot"])
    parser.add_argument("--src", required=True)
    parser.add_argument("--ann", help="COCO: one annotation json")
    parser.add_argument("--ann-glob", default="**/*.json", help="COCO: annotation files under --src (split from name)")
    parser.add_argument("--names", help="YOLO: comma-separated class names in id order (person-like ones are kept)")
    parser.add_argument("--names-from-yaml", help="YOLO: read class names from files with this name (e.g. data.yaml)")
    parser.add_argument("--name", required=True, help="output folder under data/datasets/")
    parser.add_argument("--licence", required=True, help="recorded in dataset.yaml; check it before shipping weights")
    parser.add_argument("--noncommercial", action="store_true", help="mark as evaluation/research only")
    parser.add_argument("--every", type=int, default=5, help="MOT videos: keep every Nth frame")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--keep-splits", action="store_true", help="use the source's own train/val/test split")
    parser.add_argument("--all-split", choices=["train", "val", "test"], help="put everything in one split")
    parser.add_argument("--max-negative-frac", type=float, default=0.15,
                        help="cap on images without people (hard negatives) as a fraction of the dataset")
    parser.add_argument("--group-regex", default=r"[_\-. ]?(frame)?\d+$",
                        help="removed from file stems to find the scene/sequence a frame belongs to")
    parser.add_argument("--group-depth", type=int, default=0, help="group = first N folders of the relative path")
    parser.add_argument("--min-box-px", type=float, default=4.0, help="drop boxes smaller than this (both sides)")
    parser.add_argument("--tile", type=int, default=0,
                        help="cut images whose long side exceeds 1.5x this into overlapping tiles of this size")
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--max-side", type=int, default=1920, help="downscale larger untiled images to this long side")
    parser.add_argument("--fraction", type=float, default=1.0, help="keep this share of scenes (light subsets)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    src = Path(args.src)
    readers = {"yolo": from_yolo, "coco": from_coco, "voc": from_voc, "mot": from_mot}
    items = readers[args.format](src, args)

    out = Path("data/datasets") / args.name
    if out.exists():
        shutil.rmtree(out)
    counts = defaultdict(lambda: {"images": 0, "negatives": 0, "boxes": 0})
    state = {"neg": 0, "pos": 0}
    sources = []  # (output stem, split, original file) so evaluations can join back to the source labels
    open_video, cap = None, None
    for item in items:
        if args.fraction < 1 and _unit_hash("frac|" + item.group) >= args.fraction:
            continue  # subsample whole scenes, deterministically, so reruns pick the same data
        if isinstance(item.source, tuple):  # (video, frame index)
            video, f = item.source
            if video != open_video:
                if cap is not None:
                    cap.release()
                cap, open_video = cv2.VideoCapture(str(video)), video
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
        else:
            img = cv2.imread(str(item.source))
            ok = img is not None
        if not ok:
            continue
        boxes = item.boxes
        if item.normalized:
            h, w = img.shape[:2]
            boxes = [(x1 * w, y1 * h, x2 * w, y2 * h) for x1, y1, x2, y2 in boxes]
        split = args.all_split or (args.keep_splits and item.split) or split_of(item.group, args.val, args.test)
        stem = hashlib.md5(f"{item.group}|{item.source}".encode()).hexdigest()[:16]
        for suffix, piece, piece_boxes in pieces(img, boxes, args):
            if write_piece(out, split, stem + suffix, piece, piece_boxes, args, state, counts):
                src_name = item.source if not isinstance(item.source, tuple) else f"{item.source[0]}#{item.source[1]}"
                sources.append((stem + suffix, split, Path(str(src_name)).as_posix().replace(src.as_posix() + "/", "", 1)))
    if cap is not None:
        cap.release()

    meta = {
        "path": str(out.resolve()),
        **{s: f"images/{s}" for s in ("train", "val", "test") if (out / "images" / s).exists()},
        "names": {0: "person"},
        "source": str(src),
        "licence": args.licence,
        "commercial_use": not args.noncommercial,
        "counts": {k: dict(v) for k, v in counts.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (out / "sources.csv").write_text("stem,split,source\n" + "".join(f"{s},{sp},{f}\n" for s, sp, f in sources))
    print(yaml.safe_dump({"dataset": str(out / "dataset.yaml"), "counts": meta["counts"]}, sort_keys=False))


if __name__ == "__main__":
    main()
