"""Download training datasets listed in configs/datasets.yaml and convert each to data/datasets/<name>/.

    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py --only hit_uav uav_lowalt          # download + convert
    python scripts/fetch_datasets.py --only forestpersons --convert-only  # re-convert existing download
    python scripts/fetch_datasets.py --only hit_uav --delete-raw          # free disk after converting

Hugging Face: gated sets (ForestPersons) need the terms accepted on the dataset page and a token
(`hf auth login`, or HF_TOKEN). Roboflow: set ROBOFLOW_API_KEY (free account). Downloads resume.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import prepare_dataset  # noqa: E402  (sibling script)

RAW = Path("data/external")


def fetch_hf(name: str, spec: dict) -> Path:
    from huggingface_hub import snapshot_download

    dest = RAW / name
    for attempt in range(5):  # thousands of small files: transient HTTP errors are normal, downloads resume
        try:
            snapshot_download(spec["repo"], repo_type="dataset", allow_patterns=spec.get("allow_patterns"),
                              local_dir=dest, max_workers=16)
            break
        except Exception as exc:
            if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
                raise SystemExit(f"{spec['repo']} is gated: accept its terms at https://huggingface.co/datasets/"
                                 f"{spec['repo']} and run `hf auth login` (or set HF_TOKEN).") from exc
            print(f"  download interrupted ({exc.__class__.__name__}), retry {attempt + 1}/5")
            time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit(f"download of {spec['repo']} kept failing; re-run to resume")
    for z in dest.rglob("*.zip"):
        marker = z.with_suffix(".unzipped")
        if not marker.exists():
            print(f"  unzip {z.name}")
            with zipfile.ZipFile(z) as fh:
                fh.extractall(z.parent)
            marker.touch()
    return dest


def fetch_hf_parquet(name: str, spec: dict) -> Path:
    """Parquet detection sets (image bytes + objects{category, bbox xywh}) -> YOLO folders per split."""
    import pyarrow.parquet as pq

    raw = fetch_hf(name, spec)
    out = raw / "yolo"
    if out.exists():
        return out
    keep = set(spec["person_categories"])
    ignore = set(spec.get("ignore_categories", [0]))
    import cv2

    for parquet in sorted(raw.rglob("*.parquet")):
        split = prepare_dataset.split_hint([parquet.name.split("-")[0]]) or "train"
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        table = pq.ParquetFile(parquet)
        for batch in table.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                lines = []
                objs = row["objects"]
                for cat, (x, y, bw, bh) in zip(objs["category"], objs["bbox"]):
                    if cat in ignore:  # unlabelled crowds: grey them out so they don't teach "background"
                        img[y : y + bh, x : x + bw] = 114
                    elif cat in keep and bw > 0 and bh > 0:
                        lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                cv2.imwrite(str(out / split / "images" / f"{row['id']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
                (out / split / "labels" / f"{row['id']}.txt").write_text("\n".join(lines))
        print(f"  {parquet.name} -> {out / split}")
    return out


def fetch_roboflow(name: str, spec: dict) -> Path:
    """Roboflow export via its REST API (same call the `roboflow` package makes, without its pinned deps)."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY (free account: roboflow.com -> Settings -> API keys)")
    dest = RAW / name
    if (dest / "data.yaml").exists():
        return dest
    url = (f"https://api.roboflow.com/{spec['workspace']}/{spec['project']}/{spec['version']}/"
           f"{spec.get('export_format', 'yolov11')}?api_key={key}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                info = json.load(resp)
            link = info["export"]["link"]
            break
        except Exception as exc:  # export may still be generating, or rate limited (HTTP 429)
            print(f"  waiting for export ({exc}), retry {attempt + 1}/5")
            time.sleep(20 * (attempt + 1))
    else:
        raise SystemExit(f"could not get an export link for {name}")
    print(f"  downloading {spec['workspace']}/{spec['project']} v{spec['version']}")
    with urllib.request.urlopen(link, timeout=600) as resp:
        data = resp.read()
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    return dest


FETCHERS = {"hf": fetch_hf, "hf_parquet": fetch_hf_parquet, "roboflow": fetch_roboflow}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="configs/datasets.yaml")
    parser.add_argument("--only", nargs="*", help="dataset names (default: all enabled)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--convert-only", action="store_true", help="skip downloading")
    parser.add_argument("--delete-raw", action="store_true", help="delete data/external/<name> after converting")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())["datasets"]
    if args.list:
        for name, spec in manifest.items():
            flag = " (disabled)" if spec.get("disabled") else ""
            use = "commercial OK" if spec.get("commercial_use") else "NON-COMMERCIAL"
            print(f"{name:24s} {spec['source']:10s} {use:15s} {spec.get('repo') or spec.get('project')}{flag}")
        return
    names = args.only or [n for n, s in manifest.items() if not s.get("disabled")]
    for name in names:
        spec = manifest[name]
        print(f"== {name}")
        if args.convert_only:
            src = RAW / name / "yolo" if spec["source"] == "hf_parquet" else RAW / name
        else:
            src = FETCHERS[spec["source"]](name, spec)
        prep = list(spec.get("prepare", [])) + ["--src", str(src), "--name", name, "--licence", spec["licence"]]
        if not spec.get("commercial_use", False):
            prep.append("--noncommercial")
        prepare_dataset.main(prep)
        if args.delete_raw:
            shutil.rmtree(RAW / name, ignore_errors=True)
            print(f"  deleted raw download {RAW / name}")


if __name__ == "__main__":
    main()
