# Handoff: fix false-positive heads, then full-train the screen winner

_Last updated: 2026-06-16. Author: working session (Claude). Read this top-to-bottom before resuming._

## TL;DR (what to do next, in order)

1. **Patch the preprocess merge** so background/negative images survive (today it drops them — see §4). _Code, no footage needed._
2. **Trim augmentation** in `training_service` (`copy_paste 0.3→0.1`, `mixup 0.15→0.0`). _Code, no footage needed._
3. **Capture in-domain bus footage** from the real ESP32-CAM angle, extract frames, label heads + leave furniture frames as **empty-label negatives**. _BLOCKED on capturing footage — user is doing this._
4. **Wire `bus_real` into `configs/dataset.yaml`**, re-run preprocess to rebuild `merged_dataset`, verify negatives made it in.
5. **Full-train the winner `p2head`** from scratch (pretrained transfer, NOT fine-tune). Gate it against run-7.
6. (Optional) fix the `GFLOPs = nan` reporting bug (§7).

Run everything from `train_service_cap/` with the **`train`** conda env (not base/capstone).

---

## 1. Where we are

- **Stage-1 architecture screen is DONE.** Ran `python scripts/run_experiments.py --screen`
  (640, 25% data, 40 epochs). Results in `runs/experiments/RESULTS.md`.
- **Winner: `p2head`** (CBAM backbone + an extra P2 small-object head). It swept every
  metric AND is the smallest model:

  | variant | merged recall | merged mAP50 | pass mAP50 | pass recall | params |
  |---|---|---|---|---|---|
  | **p2head** | **0.559** | **0.638** | **0.428** | **0.399** | **2.95M** |
  | baseline | 0.532 | 0.599 | 0.410 | 0.389 | 3.03M |
  | p3only | 0.532 | 0.599 | 0.411 | 0.382 | 3.01M |
  | backbone_neck | 0.530 | 0.599 | 0.417 | 0.392 | 3.05M |
  | backbone | 0.529 | 0.597 | 0.408 | 0.389 | 3.03M |

  The other four are a statistical tie — CBAM **placement** barely matters; the **P2 head** is
  what moved the needle.
- **Reference (NOT comparable — full data, 300 ep):** run-7 = merged recall 0.656 / passenger
  recall 0.895. Screen numbers are a low-fidelity ranking proxy, not a quality prediction.
- **Full-train of p2head is intentionally ON HOLD** until the FP fix below — p2head's extra
  small-object sensitivity will likely produce MORE false heads, not fewer, without negatives.

### Note on the OOM we already fixed
The first screen attempt OOM'd at batch=32 on the 16 GB RTX 5080 because the desktop
(Wallpaper Engine, browsers, Discord, etc.) was also using VRAM. Fixed by lowering
`SCREEN_OVERRIDES` batch to **16** in `src/services/experiment_service.py`. `FULL_OVERRIDES`
is still batch=32 — close GPU-heavy desktop apps before the full train, or drop it to 16/24 too.

---

## 2. The problem we're solving

**Symptom:** the model detects non-human objects (esp. **seat headrests**, also poles, bag
tops, window pillars) as person/head → inflates the passenger count.

**Root cause (data, not model):**
- **Zero negatives.** Of 39,559 train labels in `merged_dataset`, *none* are empty. Every
  training image contains a head, so the model has no signal to suppress head-like shapes.
  A headrest at bus-camera distance looks almost exactly like a head.
- **Bus domain absent from training.** `bus_real/train/images` is EMPTY (just the
  `extract_frames.py` scaffold + README). Training data is CrowdHuman dense crowds + a Roboflow
  head set; deployment is an OV3660 bus interior → domain gap.
- **Secondary — aug.** run-7 used `copy_paste=0.3 + mixup=0.15 + mosaic=1.0`, which paste/blend
  heads onto arbitrary backgrounds and can teach "heads appear anywhere," hurting precision.
  (Aug was tuned for the OLD tiny dataset; unnecessary at 40k.)

**Fix priority:** (1) in-domain background/negative images >> (2) trim aug. A quick non-retrain
band-aid is bumping `ai_service_cap/.env` `CONFIDENCE_THRESHOLD` 0.35 → ~0.45 (trims low-conf FPs
at some recall cost; cure is the data, not this).

---

## 3. Dataset facts (as of this writing)

- `merged_dataset/`: **39,668** train images / **39,559** train labels (≈109 images already
  label-less by omission, 0.3%), **5,381** val. `nc: 1`, `names: [person]`, head box style.
- Built by `src/services/preprocess_service.py` from `configs/dataset.yaml`:
  - `passenger_counting` (Roboflow, head, train/valid/test)
  - `crowdhuman` (converted via `scripts/convert_crowdhuman.py`, head, train-only — dense crowds
    deliberately kept out of val so the val yardstick stays passenger-like).
- `bus_real/`: empty scaffold. Layout `bus_real/{train,valid,test}/{images,labels}`. Images are
  git-ignored; labels are committed. `class 0 = head`, matches merged set.

---

## 4. BUG TO FIX FIRST — preprocess drops negatives

`src/services/preprocess_service.py` cannot currently carry background images into the merged set:

- `_copy_split()` and `_process_with_auto_split()` both skip any image whose label file is
  missing: `if not lbl.exists(): continue`. → images with **no** label are dropped entirely.
- `_remap_label()` returns early on empty text (`if not text: return stats`) **without writing
  an output label**. → an image with an **empty** `.txt` gets copied but no label is emitted
  downstream; stats undercount.

**Required change:** treat an *intentional empty `.txt`* as a background sample —
copy the image AND write an empty `.txt` through to `merged_dataset/<split>/labels/`.
Do NOT auto-include images that simply lack a label file (too easy to pull in unlabeled
positives by accident); require an explicit empty `.txt` to mark a negative.

Suggested approach: in the copy loops, when `lbl.exists()`, copy the image; if its text is empty,
write an empty label to the output and count it as a background (add a `backgrounds` stat). Keep
the dedup (`_sha256`) behavior. Add/adjust a unit test under `tests/` for the empty-label case.

---

## 5. Aug trim (step 2)

In `src/services/training_service.py` (the `model.train(...)` call — see run-7's `args.yaml` for
current values): set `copy_paste=0.1` (from 0.3) and `mixup=0.0` (from 0.15). Leave `mosaic`,
`hsv_*`, `fliplr`, `erasing` as-is unless FPs persist. These are passed through training; confirm
they appear in the next run's `args.yaml`.

---

## 6. Capturing + labeling bus negatives (step 3 — the blocked part)

Negatives MUST come from the **real ESP32-CAM mounting angle** (height/lens/lighting). Generic
web images are a weak stopgap only.

1. Get frames with the existing tool:
   - video: `python bus_real/extract_frames.py --video ride1.mp4 --split train --every 2`
   - capture dump: `python bus_real/extract_frames.py --images-dir ./dump --split train`
2. Label **visible heads** as positives (YOLO format, class 0).
   Tools: Label Studio / LabelImg / Roboflow (export YOLOv8, single class).
3. For **negative** frames (empty seats, headrests, poles, bag tops, windows): create an
   **empty `.txt`** next to the image in the sibling `labels/` dir. These are the whole point.
4. Target a few hundred → ~1–2k frames, weighted toward what it false-fires on. Mix full/empty
   bus, day/night, glare. Also keep a held-out `bus_real/test/` set as the honest deployment metric.

---

## 7. Merge + full train (steps 4–5)

1. Add `bus_real` to `configs/dataset.yaml` (README has the snippet):
   ```yaml
     - path: "C:/Users/blank/projects/capstone/dev/train_service_cap/bus_real"
       name: bus_real
       class_map: {0: 0}
       box_style: head
       splits: [train, valid, test]
   ```
2. Re-run preprocess to rebuild `merged_dataset` (after the §4 patch). **Verify negatives landed:**
   count empty label files in `merged_dataset/train/labels` — should be > 0 now.
3. Full-train the winner **from scratch with pretrained transfer** (NOT fine-tuning run-7 —
   standing rule, see §9):
   ```powershell
   conda activate train
   cd C:\Users\blank\projects\capstone\dev\train_service_cap
   python scripts/run_experiments.py --full p2head
   ```
   (300 epochs / full data / batch 32 via `FULL_OVERRIDES`.) It appends to
   `runs/experiments/RESULTS.md` and logs p2head vs the run-7 honest gate
   (run-7: passenger recall 0.895 / merged recall 0.656).
4. Evaluate FPs specifically: validate on the held-out `bus_real/test/` set, and eyeball
   `val_batch*_pred.jpg` for headrest hits. Precision/recall on the bus set is the real verdict.

### Optional: GFLOPs reporting bug
`_model_stats()` in `src/services/experiment_service.py` logs `GFLOPs = nan` because
`m.info(verbose=False)` isn't returning the gflops field at the expected index. Worth fixing
before committing p2head to the Pi4 — FLOPs/latency matter on ARM. Doesn't affect ranking.

---

## 8. Deploy artifacts (context — already done this session)

- `ai_service_cap/src/models/runs/detect/runs/` now has copies of runs **6 and 7** (was 2–5).
  Run-7 came with full export set: `best.pt`, `best.onnx`, `best_ncnn_model/`.
- `ai_service_cap/.env` `MODEL_PATH` was repointed from stale run-2 → **run-7** `best.onnx`.
  - On the Pi4, prefer NCNN: point `MODEL_PATH` at `passenger_yolov8-7/weights/best_ncnn_model`
    (NCNN is ~3–5× faster on ARM than ONNX). Kept `.onnx` for the dev `.env` convention.
- Deploy resolution is **640** (decided 2026-06-15). `YOLO_INPUT_SIZE=640`, `CONFIDENCE_THRESHOLD=0.35`.
- After p2head ships, repeat the copy + re-point `MODEL_PATH` to the new run.

---

## 9. Standing rules / gotchas

- **Never fine-tune the converged model on the same data.** Retrain from scratch with pretrained
  transfer. (`training_service._transfer_pretrained_weights` aligns the CBAM graph to `yolov8n.pt`
  and copies backbone+neck+box-head; attention + class head start random. `pretrained: false` in
  `args.yaml` is intentional — manual transfer already happened.) The `bus_real/README` line about
  "second-stage fine-tuning" CONFLICTS with this — ignore it; fold bus_real into the merged set
  and retrain.
- Use the **`train`** conda env. Run from `train_service_cap/`.
- Resolution **640** for both train and deploy (320 was measured to cost ~16 pts recall/mAP).
- The 5 screen variants live in `configs/experiments/`; defined in
  `experiment_service.py:VARIANTS`. `p2head` transfers backbone-only (its neck/head differ);
  the rest transfer `full`.
- Screen retrains the baseline too (deliberate — run-7's full numbers aren't comparable to a
  40-ep/25% screen).

---

## 10. Open decision left for the user

When footage is sorted, confirm: capture real bus footage now, or fall back to generic
public bus-interior negatives as a stopgap for a first pass? (In-domain strongly preferred.)
Also decide whether to apply the `CONFIDENCE_THRESHOLD 0.35→0.45` band-aid on the deployed
run-7 in the meantime.
