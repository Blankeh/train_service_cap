# Bus Passenger Counting — Model Training Service

Training pipeline that produces the YOLOv8 person/head-detection model deployed
by its sibling project, [`ai_service_cap`](../ai_service_cap). It merges several
raw YOLO datasets into one clean dataset, trains a CBAM-attention YOLOv8n,
exports a Raspberry Pi 4–optimized model (ONNX / NCNN), evaluates it on a
held-out test split, and optionally pushes it to a Cloudflare Worker for the Pi
to pull over the air.

Runs on a GPU workstation (developed against an RTX 5080, 16 GB). The deployment
target is a **Raspberry Pi 4 (ARM Cortex-A72, no GPU)**, which is why training
resolution and export format are tuned for cheap CPU inference.

---

## Pipeline at a glance

```
 configs/dataset.yaml ─▶ Stage 1  Preprocess  ─▶ merged_dataset/ (+ data.yaml)
                         Stage 2  Train        ─▶ runs/.../best.pt
                         Stage 3  Export       ─▶ runs/.../best.onnx (+ NCNN)
                         Stage 4  Evaluate     ─▶ mAP50 / mAP50-95 / P / R on test split
                         Stage 5  Deploy (opt) ─▶ Cloudflare Worker → Pi 4 OTA
```

Each stage is an independent service under `src/services/`, orchestrated by
`src/main.py`. A full, stage-by-stage reference — including every config field
and the geometric box-style conversions — lives in [`PIPELINE.md`](PIPELINE.md).

| Stage | Service | What it does |
|---|---|---|
| 1 Preprocess | `PreprocessService` | Merge datasets, dedup by SHA-256, polygon→bbox, head/body box-style normalization, class remap, auto train/val/test split |
| 2 Train | `TrainingService` | Build the (CBAM) YOLOv8n graph, transfer pretrained backbone/neck/box-head weights, train per `configs/training.yaml` |
| 3 Export | `ExportService` | Convert `best.pt` → ONNX (and NCNN for ARM); falls back to ONNX if NCNN tooling is missing |
| 4 Evaluate | `EvaluateService` | `model.val(split="test")` — never touches train/valid |
| 5 Deploy | `DeployService` | Multipart-upload the model to a Cloudflare Worker (R2 + version bump) |

---

## Architecture: CBAM + P2 small-object head

The default architecture is **YOLOv8n with CBAM attention blocks** inserted after
each detection-scale C2f (`configs/yolov8n_cbam.yaml`). Because those insertions
shift every later layer index, `TrainingService` does a **positional** weight
transfer from `yolov8n.pt` — walking both module lists and skipping the
custom-only attention layers — so the backbone, neck, and Detect box-head all
start pretrained while only the attention blocks and class head start random.
This is registered via `src/core/custom_modules.py` (`CBAM`, `ECA`, P2 modules).

### Experiment harness

`src/experiments.py` (run via `python -m src.experiments` or
`scripts/run_experiments.py`) runs an **architecture ablation** over five
variants — CBAM placement (`baseline`, `p3only`, `backbone`, `backbone_neck`)
plus an extra **P2 small-object head** (`p2head`):

```powershell
python -m src.experiments --screen          # cheap ranking pass over all variants
python -m src.experiments --full p2head     # full-train one winner from scratch
```

A `--screen` pass trains every variant under the same cheap budget (640 px, 25%
data, 40 epochs) and ranks them; `--full` trains the chosen winner at the full
budget (300 epochs, full data). Results append to `runs/experiments/RESULTS.md`.

> **Current status (handoff):** the screen is **done** — `p2head` won every
> metric and is the smallest model. Its full train is **on hold** pending a
> false-positive fix (the model fires on seat headrests because the merged set
> has zero negative/background images). The plan to add in-domain bus negatives
> and retrain is documented in [`NEXT_STEPS_fp_fix.md`](NEXT_STEPS_fp_fix.md).

> **Standing rule:** never fine-tune the converged model on the same data —
> retrain from scratch with pretrained transfer. New data (e.g. `bus_real/`) is
> folded into the merged set and the model is retrained, not incrementally tuned.

---

## Requirements

- Python 3.10+ with an NVIDIA GPU + CUDA for training
- Use the **`train`** conda environment for all commands (not `base`/`capstone`)
- See [`requirements.txt`](requirements.txt) — Ultralytics, OpenCV, ONNX /
  ONNX Runtime, PyYAML, httpx. NCNN/OpenVINO export tools are optional.

## Setup

```powershell
conda activate train
cd train_service_cap
pip install -r requirements.txt
```

## Running

```powershell
python -m src.main                    # full pipeline: preprocess → train → export → evaluate
python -m src.main --skip-preprocess  # reuse an existing merged_dataset/
python -m src.main --deploy           # also push the model to Cloudflare after evaluate
```

(`scripts/run_pipeline.py` is a thin wrapper that runs the same `src.main` module
from the project root.)

---

## Configuration

| File | Purpose |
|---|---|
| [`configs/dataset.yaml`](configs/dataset.yaml) | Source datasets, per-dataset `box_style`/`class_map`/`splits`, output classes, target box style, dedup seed |
| [`configs/training.yaml`](configs/training.yaml) | Architecture (`cbam`/`default`), epochs/batch/imgsz/patience/device, export format, deploy block |
| [`configs/yolov8n_cbam.yaml`](configs/yolov8n_cbam.yaml) | The CBAM YOLOv8n model graph |
| `configs/experiments/*.yaml` | Per-variant model graphs + the `passenger_eval` continuity set |
| `.env` | Cloudflare Worker URL / API key, `CLOUDFLARE_DEPLOY_ENABLED`, Roboflow key |

Key `training.yaml` knobs: `epochs: 300`, `batch: 32` (RTX 5080 16 GB),
`imgsz: 640` (320 was measured to cost ~16 pts recall/mAP), `device: "0"`, and
an `export.format` of `ncnn` (fastest on ARM NEON) with automatic ONNX fallback.

### Deploy (optional)

Disabled by default. Enable with `--deploy`, or set `deploy.enabled: true` in
`training.yaml`, or export the env vars:

```powershell
$env:CLOUDFLARE_DEPLOY_ENABLED = "true"
$env:CLOUDFLARE_WORKER_URL     = "https://your-worker.workers.dev"
$env:CLOUDFLARE_API_KEY        = "your_api_key"
python -m src.main
```

The Worker stores the model in R2 and bumps a version number; the Pi polls
`GET /api/model/metadata` and downloads when it sees a new version.

---

## Datasets

`merged_dataset/` is built from the sources in `configs/dataset.yaml`
(a Roboflow passenger-counting head set + CrowdHuman dense crowds, train-only).
All boxes are normalized to a single `person` head-style class.

- `scripts/convert_crowdhuman.py` converts CrowdHuman ODGT + zipped images into a
  YOLO head-detection dataset.
- `bus_real/` is a scaffold for **in-domain** bus footage captured from the real
  ESP32-CAM angle. `bus_real/extract_frames.py` pulls frames from a video or an
  image dump; images are git-ignored, labels are committed. Negative/background
  frames (headrests, poles, empty seats) get an **empty** `.txt` and are the key
  to fixing the false-positive problem above.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/run_pipeline.py` | Run the full training pipeline (`src.main`) from the project root |
| `scripts/run_experiments.py` | Run the architecture ablation (`src.experiments`) |
| `scripts/convert_crowdhuman.py` | Convert CrowdHuman → YOLO head dataset |
| `scripts/export_evaluate.py` | Export + evaluate an existing checkpoint without retraining |
| `scripts/deploy_model.py` | Upload a given `.onnx`/`.pt` to the Cloudflare Worker |

## Docker

A GPU-enabled image is provided for reproducible training:

```bash
docker compose up --build      # requires nvidia-container-toolkit on the host
```

`docker-compose.yml` mounts the raw datasets read-only and persists
`merged_dataset/` and `runs/` across container restarts.

## Testing

```powershell
pytest
```

Covers the preprocess, training, export, evaluate, and deploy services.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See
[`LICENSE`](LICENSE) for the full text.
