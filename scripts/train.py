"""Fine-tune the person detector on one or more prepared datasets (see prepare_dataset.py).

    python scripts/train.py --data nomad heridal own_clips --name grass_v1
    # generalisation check: train without a source, validate only on it (simulates an unseen client clip)
    python scripts/train.py --data nomad heridal --val-on own_clips --name grass_v1_holdout
    python scripts/train.py --config configs/train_laptop.yaml ... --resume   # continue after an interruption

Datasets marked non-commercial are refused unless --allow-noncommercial is given, and the resulting
run is then labelled research-only, so such weights cannot ship by accident.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from grassdet.augment import build_augmentations


def load_meta(name: str) -> dict:
    path = Path(name) if name.endswith(".yaml") else Path("data/datasets") / name / "dataset.yaml"
    meta = yaml.safe_load(path.read_text())
    meta["_root"] = path.parent.resolve()  # not meta["path"]: datasets may be converted on another machine
    return meta


def split_dirs(metas: list[dict], split: str) -> list[str]:
    return [str(m["_root"] / m[split]) for m in metas if split in m]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="dataset names under data/datasets/ (or yaml paths)")
    parser.add_argument("--val-on", nargs="*", default=[], help="validate only on these datasets (held out of training)")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--project", default="runs/train", help="output folder")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from its last.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override ultralytics args")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_metas = [load_meta(n) for n in args.data]
    val_metas = [load_meta(n) for n in args.val_on] or train_metas
    restricted = [m["source"] for m in train_metas + val_metas if not m.get("commercial_use", True)]
    if restricted and not args.allow_noncommercial:
        raise SystemExit(f"non-commercial datasets in use: {restricted}. Pass --allow-noncommercial for research runs.")

    train_dirs = split_dirs(train_metas, "train")
    val_dirs = split_dirs(val_metas, "val") + (split_dirs(val_metas, "train") if args.val_on else [])
    if not train_dirs or not val_dirs:
        raise SystemExit("need at least one train split and one val split")
    run_dir = Path(args.project) / args.name
    last = run_dir / "weights" / "last.pt"
    train_args = dict(cfg.get("ultralytics", {}))
    for item in args.set:
        key, value = item.split("=", 1)
        train_args[key] = yaml.safe_load(value)
    if args.resume and last.exists():
        # Runs get interrupted (laptop out of RAM, paused by the user); Ultralytics restores epoch, optimizer and schedule.
        # The augmentation objects aren't stored in the checkpoint, so pass them again, along with the few
        # settings Ultralytics allows to change on resume (so a run can continue with less memory).
        resumable = {k: v for k, v in train_args.items()
                     if k in ("imgsz", "batch", "device", "close_mosaic", "save_period", "workers", "cache", "patience")}
        print(f"resuming {last} with {resumable}")
        YOLO(str(last)).train(resume=True, augmentations=build_augmentations(cfg.get("extra_augmentations", 1.0)),
                              **resumable)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "train": train_dirs,
        "val": val_dirs,
        "names": {0: "person"},
        "licences": {m["source"]: m["licence"] for m in train_metas},
        "research_only": bool(restricted),
    }, sort_keys=False))

    model = YOLO(cfg["model"])
    model.train(
        data=str(data_yaml),
        project=str(run_dir.parent.resolve()),
        save_period=train_args.pop("save_period", 5),
        name=args.name,
        exist_ok=True,
        augmentations=build_augmentations(cfg.get("extra_augmentations", 1.0)),
        **train_args,
    )
    print(f"\nweights: {run_dir / 'weights' / 'best.pt'}" + ("  (RESEARCH ONLY: non-commercial data)" if restricted else ""))


if __name__ == "__main__":
    main()
