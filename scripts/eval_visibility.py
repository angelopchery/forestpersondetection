"""Detection recall on ForestPersons by how visible the person is (20/40/70/100 %) and by pose.

Answers the client's real question ("how hidden can someone be and still be found?") instead of one
mAP number. Uses the official test split, joined back to the original ForestPersons labels through
data/datasets/<name>/sources.csv.

    python scripts/eval_visibility.py --weights runs/train/grass_v1/weights/best.pt
    python scripts/eval_visibility.py --weights models/yolo26s.pt --name baseline   # COCO model for comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_annotations(raw: Path, repo: str, split: str) -> dict:
    local = raw / "annotations" / f"{split}.json"
    if local.exists():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download  # raw download deleted: the 6 MB zip is enough

    with zipfile.ZipFile(hf_hub_download(repo, "annotations.zip", repo_type="dataset")) as z:
        return json.load(z.open(f"annotations/{split}.json"))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="forestpersons", help="name under data/datasets/")
    parser.add_argument("--repo", default="etri/ForestPersons")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a detection as correct")
    parser.add_argument("--name", help="label for the results file (default: weights folder name)")
    parser.add_argument("--out", default="outputs/eval")
    args = parser.parse_args()

    root = Path("data/datasets") / args.dataset
    # join on the basename: sources.csv is relative to the download folder (images/<seq>/<frame>.jpg), the
    # annotation file_name to the images folder (<seq>/<frame>.jpg); frame names are unique in ForestPersons
    sources = {r["stem"]: Path(r["source"]).name for r in csv.DictReader((root / "sources.csv").open())
               if r["split"] == args.split}
    data = load_annotations(Path("data/external") / args.dataset, args.repo, args.split)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_file = {Path(im["file_name"]).name: im for im in data["images"]}
    gt = defaultdict(list)  # image id -> [(x1, y1, x2, y2, pose, visibility)]
    for a in data["annotations"]:
        pose, vis = cats[a["category_id"]].rsplit("_", 1)
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((x, y, x + w, y + h, pose, int(vis)))

    model = YOLO(args.weights)
    found, total = defaultdict(int), defaultdict(int)
    false_pos, images = 0, 0
    paths = [str(p) for p in sorted((root / "images" / args.split).glob("*.jpg"))]
    chunk = 4  # ultralytics 8.4 runs a list source as ONE batch (236 images = 9 GB): feed it a few at a time
    for result in (r for i in range(0, len(paths), chunk)
                   for r in model.predict(paths[i:i + chunk], imgsz=args.imgsz, conf=args.conf, classes=[0], verbose=False)):
        stem = Path(result.path).stem
        im = by_file.get(sources.get(stem, ""))
        if im is None:
            continue
        images += 1
        h, w = result.orig_shape
        scale = w / im["width"]  # converted images may be downscaled (--max-side)
        boxes = [(x1, y1, x2, y2, pose, vis) for x1, y1, x2, y2, pose, vis in gt[im["id"]] if vis > 0]
        g = np.array([b[:4] for b in boxes], dtype=float).reshape(-1, 4) * scale
        p = result.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        matched = set()
        if len(g) and len(p):
            m = iou_matrix(g, p)
            for pi in np.argsort(-result.boxes.conf.cpu().numpy()):  # greedy, best predictions first
                gi = int(np.argmax(m[:, pi]))
                if m[gi, pi] >= args.iou and gi not in matched:
                    matched.add(gi)
        false_pos += len(p) - len(matched)
        for gi, b in enumerate(boxes):
            for key in (("visibility", b[5]), ("pose", b[4]), ("all", "all")):
                total[key] += 1
                found[key] += gi in matched

    def table(kind):
        rows = sorted(k for k in total if k[0] == kind)
        return {str(k[1]): {"recall": round(found[k] / total[k], 3), "persons": total[k]} for k in rows}

    results = {"weights": args.weights, "images": images, "conf": args.conf, "iou": args.iou,
               "recall": table("all")["all"]["recall"], "false_positives_per_image": round(false_pos / max(images, 1), 3),
               "by_visibility": table("visibility"), "by_pose": table("pose")}
    print(f"\n{args.weights}  ({images} test images, conf>={args.conf}, IoU>={args.iou})")
    print(f"  overall recall {results['recall']:.3f}   false positives/image {results['false_positives_per_image']:.2f}")
    for kind in ("by_visibility", "by_pose"):
        print("  " + kind.replace("_", " ") + ":  " + "   ".join(f"{k}: {v['recall']:.2f} (n={v['persons']})"
                                                               for k, v in results[kind].items()))
    out = Path(args.out) / f"visibility_{args.name or Path(args.weights).parent.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"  written {out}")


if __name__ == "__main__":
    main()
