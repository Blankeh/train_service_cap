# Train Service — Pipeline Reference

## How to run

```powershell
cd train_service
python -m src.main                   # full pipeline (preprocess → train → export → evaluate)
python -m src.main --skip-preprocess # skip stage 1 if merged_dataset already exists
python -m src.main --deploy          # also push model to Cloudflare after evaluate
```

---

## Stage 1 — Preprocess (`PreprocessService`)

**What it does:** Merges multiple raw YOLO datasets into a single `merged_dataset/` folder with clean train/valid/test splits.

**Steps:**
1. Iterates each dataset entry in `configs/dataset.yaml`
2. For each image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`):
   - Skips images that have no matching label file
   - Computes a SHA-256 hash of the image bytes — skips exact duplicates across datasets
3. **Class remapping** — rewrites label files so all class IDs map to the unified class list (e.g. `man→0`, `woman→0`, `head→0` all become `person`)
4. **Auto-split** — if a dataset only has a `train/` folder, it shuffles the unique images (seeded for reproducibility) and splits them into train/valid/test according to the ratios in `auto_split`
5. Writes `merged_dataset/data.yaml` — the YOLO dataset descriptor consumed by training

**Output:**
```
merged_dataset/
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/
  data.yaml
```

**Config:** `configs/dataset.yaml`

---

## Stage 2 — Train (`TrainingService`)

**What it does:** Fine-tunes a YOLOv8 model on the merged dataset.

**Steps:**
1. Loads the base model (`yolov8n.pt`) from Ultralytics — downloads automatically if not present
2. Calls `model.train(...)` with the params from `configs/training.yaml`
3. Ultralytics saves checkpoints to `runs/passenger_yolov8/weights/`

**Key settings (`configs/training.yaml`):**

| Setting | Value | Notes |
|---------|-------|-------|
| `base_model` | `yolov8n.pt` | Nano — smallest, best for Pi4 edge deployment |
| `epochs` | 100 | Max epochs; early stopping will cut this short |
| `patience` | 20 | Stops if mAP doesn't improve for 20 consecutive epochs |
| `batch` | 64 | Tuned for RTX 5080 (16 GB VRAM) at imgsz 640 |
| `imgsz` | 640 | Input resolution |
| `device` | `"0"` | RTX 5080, CUDA device 0 |
| `workers` | 8 | Dataloader workers |

**Output:** `runs/passenger_yolov8/weights/best.pt`

---

## Stage 3 — Export (`ExportService`)

**What it does:** Converts `best.pt` (PyTorch) to `best.onnx` for deployment on the Raspberry Pi 4.

**Why ONNX:** The Pi 4's ARM Cortex-A72 has no GPU. ONNX Runtime gives the best CPU inference throughput and is compatible with the Ultralytics pipeline on the Pi.

**Steps:**
1. Loads `best.pt` with Ultralytics
2. Calls `model.export(format="onnx", simplify=True, opset=17, dynamic=False)`
3. `simplify=True` prunes redundant ops — smaller file, faster inference
4. `dynamic=False` fixes the batch dimension to 1 for edge deployment

**Output:** `runs/passenger_yolov8/weights/best.onnx`

---

## Stage 4 — Evaluate (`EvaluateService`)

**What it does:** Runs the exported ONNX model against the held-out **test split** and reports detection metrics.

**Steps:**
1. Loads `best.onnx` with Ultralytics
2. Calls `model.val(split="test", ...)` — never touches train or valid data
3. Rounds and logs four metrics

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| `mAP50` | Mean average precision at IoU threshold 0.50 |
| `mAP50-95` | mAP averaged over IoU thresholds 0.50–0.95 (stricter) |
| `precision` | Of all predicted boxes, fraction that are correct |
| `recall` | Of all ground-truth boxes, fraction that were found |

---

## Stage 5 — Deploy (`DeployService`) — optional

**What it does:** Uploads `best.onnx` to a Cloudflare Worker so the Pi 4 can pull it without a direct network connection to your machine.

**Flow:**
1. Validates that `worker_url` and `api_key` are set (fails fast if not)
2. Strips trailing `/` from the URL and posts to `POST /api/model/upload` as multipart form-data
3. Sends `Authorization: Bearer <api_key>` header
4. On success the Worker stores the model in R2 and bumps the version number
5. The ai_service on the Pi polls `GET /api/model/metadata` and downloads when it sees a new version

**Enable deploy:**
```powershell
# Option A — flag
python -m src.main --deploy

# Option B — env vars
$env:CLOUDFLARE_DEPLOY_ENABLED = "true"
$env:CLOUDFLARE_WORKER_URL     = "https://your-worker.workers.dev"
$env:CLOUDFLARE_API_KEY        = "your_api_key"
python -m src.main
```

---

## Output summary

| File | Description |
|------|-------------|
| `merged_dataset/data.yaml` | YOLO dataset descriptor |
| `runs/passenger_yolov8/weights/best.pt` | Best PyTorch checkpoint |
| `runs/passenger_yolov8/weights/best.onnx` | Exported model for Pi 4 |
| `runs/passenger_yolov8/results.csv` | Per-epoch training metrics |
| `runs/passenger_yolov8/plots/` | Confusion matrix, PR curve, etc. |
