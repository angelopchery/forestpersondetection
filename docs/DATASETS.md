# Datasets

No footage from the client was available and none could be recorded, so the detector is trained on a
deliberately small mix of public datasets chosen for diversity of viewpoint, spectrum and person size,
not volume. Everything is described in one manifest, [`configs/datasets.yaml`](../configs/datasets.yaml),
and rebuilt by script, so the training data is reproducible from scratch.

## What grass_v1 was trained on

Image counts are after conversion, splitting and de-duplication, exactly as seen by training.

| Source (manifest name) | Why it is in the mix | Train | Val | Test | Boxes (train) | Licence |
|---|---|---|---|---|---|---|
| **ForestPersons** `forestpersons` ([HF, gated](https://huggingface.co/datasets/etri/ForestPersons)) | **Core.** People standing, sitting and **lying** in forest and vegetation, filmed from 1.5–2 m, with 20 / 40 / 70 / 100 % visibility labels. The closest public match to the task. Every 50th frame, official splits. | 1,439 | 387 | 235 | 3,088 | CC BY-NC-SA 4.0 + no weapons / mass-surveillance use |
| **HIT-UAV** `hit_uav` ([HF](https://huggingface.co/datasets/Kiuyha/hit-uav-thermal-human-detection)) | Thermal drone images of people in fields and on roads | 1,163 | 415 | 205 | 6,910 | CC BY 4.0 |
| **Low-altitude UAV** `uav_lowalt` ([HF](https://huggingface.co/datasets/Mikiee/Person_Detection_and_Re-Identification_from_Low_Altitude_UAV-based_platform)) | Low drone RGB, people outdoors | 240 | – | – | 326 | CC BY 4.0 |
| **VisDrone 2019-DET** `visdrone_lite` ([HF](https://huggingface.co/datasets/shijli/visdrone2019-det)) | Tiny, distant people; tiled to 960 px so they keep their pixels. 25 % of the train tiles (mostly city scenes, so less is enough) | 1,475 | 326 | – | 16,361 | CC BY-NC-SA 3.0 |
| **Total** | | **4,317** | **1,128** | 440 | 26,685 | research-only (non-commercial sources included) |

Disk: about 2.2 GB converted; the raw ForestPersons subset (1.1 GB) can be deleted after conversion.

**Not used (yet):** ForestPersonsIR (same scenes in thermal): access requested, still pending. The four
Roboflow sets in the manifest are disabled until each page is checked by hand for licence and origin.

## Preparation pipeline

```
fetch_datasets.py ──► data/external/<name>/      (raw download, only the selected files)
        │
        ▼ prepare_dataset.py
data/datasets/<name>/  images/{train,val,test}  labels/…  dataset.yaml (licence, counts)  sources.csv
        │
        ├─► subset_dataset.py   visdrone → visdrone_lite (25 % of train, hard links: no extra disk)
        └─► dedupe_datasets.py  perceptual-hash de-duplication across all sources
```

`prepare_dataset.py` reads YOLO, COCO, Pascal VOC or MOT-labelled videos and always produces the same
thing: one class `person`, YOLO labels, the licence recorded next to the data, and a `sources.csv`
mapping each image back to its original file (used by `eval_visibility.py`).

Rules that make validation scores trustworthy for unseen clips:

- **Split by scene, never by frame.** Consecutive frames are near-duplicates; a per-frame split leaks
  them into validation. Scene IDs come from folder names and frame-number suffixes. Official splits are
  kept where they exist (ForestPersons, HIT-UAV, VisDrone).
- **Cross-split de-duplication.** A 64-bit difference hash finds train images that are near-copies of a
  val/test image and removes them from train (on the HIT-UAV Roboflow export this caught 132 train images
  copied from its official test set). For static-camera sources such as ForestPersons, near-duplicates
  *within* train are kept (`--cross-split-only`): the hash cannot see a small person moving across an
  otherwise identical scene, so those frames are real training signal.
- **Hard negatives capped** at 15 % person-free images per source.
- **Tiling** of large drone images (`--tile 960`), keeping a box cut by a tile border only if ≥ 60 % of it
  is inside the tile.

### Dataset-specific findings

- **ForestPersons class names are not "person".** Its 13 COCO categories are `pose_visibility`
  combinations (`sitting_100`, `lying_40`, …) under the supercategory `person`. The converter matches on
  the supercategory. The first conversion matched nothing and silently produced one empty image; it now
  stops with an error when a file has no person categories.
- **`unknown_0` = a fully hidden person (0 % visible).** Such a box is unlearnable, and an image holding
  one is not a true negative either, so those images are skipped entirely.
- **Every 50th frame** of ForestPersons keeps every sequence at about 2 % of the 47.6 GB full download.
- **Gated access:** ForestPersons and ForestPersonsIR require accepting terms on Hugging Face with the
  same account as the token used (`hf auth login`).

## Licences

| Bucket | Sources | Consequence |
|---|---|---|
| Commercial use allowed (as stated by the source) | HIT-UAV, low-altitude UAV | Can train shippable weights |
| **Non-commercial** | ForestPersons (+ use restrictions), VisDrone | Weights are **research / demo only**; `train.py` refuses them without `--allow-noncommercial` |

Re-check every licence at the source before any commercial use.

## Researched but not used

Kept for the next iterations; licences as stated by the sources in October 2026.

| Dataset | What it adds | Licence | Why not now |
|---|---|---|---|
| ForestPersonsIR | Same scenes as ForestPersons, thermal | CC BY-NC-SA 4.0 | Access pending |
| WiSARD | Wilderness search-and-rescue, RGB + thermal | MIT-style | 40 GB, not on HF |
| NOMAD | Walking / lying / hiding actors from drones, 10 visibility levels | CC BY 4.0 (paper) | Google Drive only; licence unconfirmed on repo |
| SARD | Actors lying in grass, crouching | not stated | IEEE DataPort login |
| HERIDAL | Wilderness 4000×3000 + person-free patches | CC BY 3.0 | Site unreachable during research |
| POP | Thermal drone, people occluded by trees | CC BY 4.0 | Thermal-only; candidate for grass_v2 |
| BIRDSAI, Snapshot Serengeti | Animals in bush/grass: hard negatives | CDLA-Permissive | Next iteration |
| CrowdHuman, COD10K, LLVIP, NII-CU, Okutama | Occlusion, camouflage, thermal pairs, lying actions | Non-commercial | Licence or scope |

## Gaps no public dataset fills

- **No public dataset labels crawling.** Crawling is recognised by the behaviour stage (posture + motion),
  not learned as a class.
- **No public dataset has people crawling in grass, from ground level, with tracking IDs.** Tracking is
  therefore measured on synthetic scenes; a few labelled real clips are the first item on the roadmap.
