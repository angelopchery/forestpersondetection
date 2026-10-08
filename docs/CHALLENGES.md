# Challenges, constraints and shortcomings

This document records the conditions the project was built under, the problems met along the way and
how each was handled, and the shortcomings that remain. It is meant to be read alongside the numbers in
[RESULTS.md](RESULTS.md).

## 1. Constraints

| Constraint | Consequence |
|---|---|
| **No footage or data from the client**, and no answers to the scoping questions (camera, viewpoint, resolution, day/night, live or offline) | Nothing was available to train or test on the actual use case; every design choice had to hold for a wide range of plausible footage |
| **No possibility to record staged footage** (e.g. people crawling in grass) | Training and evaluation rely on public datasets and synthetic scenes |
| **No cloud compute.** All data preparation, training and evaluation ran locally on the developer's laptop | Hardware below sets every budget in the project |
| Laptop: NVIDIA RTX 3050 Laptop GPU (**4 GB VRAM**), **16 GB RAM**, ~**22 GB** free disk, Windows 11, shared with day-to-day use | Light dataset mix (~2 GB on disk), 768 px training with batch 6, one dataloader worker, training run in interruptible chunks |

## 2. A generalised model, by design

**`grass_v1` is a general-purpose detector for people in vegetation, not a model trained for one specific
deployment.** With no client footage, it was trained to cope with as many conditions as possible instead
of being specialised for one:

- **viewpoints:** ground level (ForestPersons), low drone (low-altitude UAV), high drone (VisDrone);
- **spectra:** colour and thermal (HIT-UAV), plus greyscale augmentation;
- **person sizes:** from ~10 px (tiled drone images) to full-frame;
- **poses:** standing, sitting, lying; partly hidden at 20–100 % visibility;
- **camera quality:** compression, blur, noise, low resolution, low light and foreground-grass occlusion
  applied as augmentations.

The trade-off is deliberate: a generalised model is reasonably good across many kinds of footage but
not the best possible on any single one. **Performance on a specific use case would increase by
fine-tuning on that use case's footage.** In practice a few hundred labelled frames from the client's own
cameras and sites, plus tuning the confidence and tracking thresholds on a handful of labelled clips, is
the usual way to close most of that gap. The pipeline, training scripts and evaluation tools are all in
place for that step (`prepare_dataset.py` accepts labelled videos in MOT format; `evaluate.py` scores the
result).

## 3. Challenges met during development

| Challenge | What happened | How it was handled |
|---|---|---|
| **Dataset size vs. disk** | ForestPersons, the best-matching dataset, is 47.6 GB; ~22 GB of disk was free | Download only every 50th frame (1.1 GB raw, 0.5 GB converted): consecutive frames are near-duplicates, so this keeps the scene diversity at ~2 % of the size |
| **Gated datasets** | ForestPersons and ForestPersonsIR require accepting terms per account | ForestPersons approved and used; **ForestPersonsIR still pending**, so thermal coverage is limited (see §4) |
| **Download throttling** | Hugging Face returned HTTP 429 (rate limit) on datasets with thousands of small files | Downloads retry and resume automatically |
| **Undocumented label format** | ForestPersons categories are named `sitting_100`, `lying_40`, … rather than `person`; the first conversion matched nothing and silently produced one empty image | Match on the COCO supercategory; the converter now stops with an error instead of producing an empty dataset |
| **Fully hidden people** | ForestPersons labels 0 %-visible people (`unknown_0`); used as boxes they teach hallucination, dropped they turn images into false "no person" examples | Images containing them are skipped |
| **De-duplication too aggressive** | The near-duplicate filter removed 505 of 1,446 ForestPersons training frames: a small person moving across a static scene does not change the image hash | New `--cross-split-only` mode: only remove train images that leak into val/test (1 found); keep within-train near-duplicates |
| **Config parsing bug** | A dataset option whose value starts with `-` was read as a new flag and stopped the download script | Fixed in the manifest (`--group-regex=…` form) |
| **4 GB of VRAM** | Ultralytics' automatic batch sizing chose batch 1 on this GPU; batch 8 at 768 px used 3.45 of 4 GB | Fixed batch 6 at 768 px (measured), AMP on; inference evaluation fed in chunks after a 9 GB allocation attempt (Ultralytics batches a whole image list at once) |
| **16 GB of RAM** | The first training run was killed after 2 epochs when training (two dataloader worker processes) plus everyday apps exhausted memory | One dataloader worker; `train.py` now re-applies memory settings when resuming (previously the checkpoint's settings silently won) |
| **Power limits** | On battery the GPU is capped at 30 W | Long runs on AC power: ~5 min per epoch |
| **Shared machine** | The laptop was needed for other work during training | Checkpoints every 2 epochs and `--resume`: grass_v1 was trained in three chunks with no loss beyond the epoch in progress |
| **No crawling data anywhere** | No public dataset labels crawling | Crawling recognised by a separate behaviour stage (pose + camera-compensated motion) instead of a detector class |
| **No tracking ground truth for the task** | Public tracking sets are street or drone footage | Synthetic grass scenes with exact ground truth for regression testing; MOT-format evaluator ready for real labelled clips |
| **Licensing** | The best-matching data is non-commercial; Ultralytics is AGPL-3.0 | Non-commercial data is blocked from training unless explicitly allowed, and such runs are labelled research-only; code published under AGPL-3.0; commercial path documented |
| **Newer tracking methods** | TAPNext / TAPNext++ was considered as a modern addition | Measured on this laptop: 2–4× the cost of the whole pipeline per frame, and it cannot find people (a point tracker needs a detector). Kept as an optional experiment, not adopted ([ARCHITECTURE.md](ARCHITECTURE.md#tapnext--tapnext-google-deepmind-tapnet-evaluated-not-adopted)) |

## 4. Shortcomings

Ranked by how likely they are to matter on real deployment footage. Evidence for each is in
[RESULTS.md](RESULTS.md).

1. **Not validated on the target use case.** Results are on public test sets and synthetic scenes. This is
   the direct consequence of having no client footage, and the main reason a use-case fine-tune (§2) is
   the first recommended step.
2. **Crawling is unvalidated on real people.** It works on synthetic clips; no real crawling footage was
   available to measure it.
3. **Identity breaks under fast camera motion.** On a 4K drone clip, one hiker was detected throughout but
   split into three track IDs while the camera descended and turned.
4. **Thermal and night footage.** A top-down, white-hot thermal clip with a HUD overlay produced no tracks.
   The quality report flagged it for manual review, but the detection itself failed; thermal training data
   is limited to HIT-UAV until ForestPersonsIR access is granted.
5. **More false positives than the stock model** at the detector level (0.07 → 0.29 per image on
   ForestPersons); the tracking filters remove most but not all.
6. **Posture is unreliable from directly above**, where "upright" is undefined; activity is often
   `unknown` on top-down drone footage.
7. **4K footage is slow** on this laptop (tiling: ~13× real time end to end).
8. **Research-only weights**, because two training sources are non-commercial.

## 5. What would remove them

| Shortcoming | Remedy | Needs |
|---|---|---|
| 1, 5 | Fine-tune on labelled frames from the client's cameras; tune thresholds on labelled clips | Client footage (a few hundred frames, a handful of clips) |
| 2 | Measure crawling precision/recall on real crawling clips | Any real crawling footage |
| 3 | Tracklet stitching after tracking (join a track that ends with one that starts close by within ~1 s) | Small code change, validated with `evaluate.py` |
| 4 | Add ForestPersonsIR and other thermal sets; white-hot/black-hot polarity augmentation; retrain | Dataset access; ~3 h of laptop training |
| 6 | Use box-shape and motion cues for top-down views; label a few drone clips | Labelled drone footage |
| 7 | TensorRT FP16 export; larger tiles or a lower tiling threshold | Engineering only |
| 8 | Retrain on commercially licensed data only, and license or replace Ultralytics | Licensing decision |
