from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


# ── Dataset config (configs/dataset.yaml) ─────────────────────────────────────

class DatasetEntry(BaseModel):
    path: str
    name: str
    class_map: dict[int, int]
    box_style: str = "head"   # "head" or "body" — what the raw labels annotate
    splits: list[str]
    auto_split: Optional[dict[str, float]] = None


class DatasetOutputConfig(BaseModel):
    path: str = "./merged_dataset"
    classes: list[str] = ["person"]
    target_box_style: str = "head"  # normalize all labels to this style
    seed: int = 42


class DatasetConfig(BaseModel):
    datasets: list[DatasetEntry]
    output: DatasetOutputConfig


# ── Training config (configs/training.yaml) ───────────────────────────────────

class ExportConfig(BaseModel):
    # Supported formats: onnx | ncnn | openvino
    # ncnn is recommended for Raspberry Pi 4 (ARM NEON, no GPU)
    # onnx is the reliable fallback when ncnn tools are unavailable
    format: str = "ncnn"
    simplify: bool = True
    opset: int = 17
    int8: bool = False       # INT8 quantization: ~2× faster on Pi4, needs calibration images
    half: bool = False       # FP16: not useful on Pi4 ARM (no FP16 SIMD)


class DeployConfig(BaseModel):
    enabled: bool = False      # opt-in via CLOUDFLARE_DEPLOY_ENABLED=true
    worker_url: str = ""       # e.g. https://your-worker.workers.dev
    api_key: str = ""
    timeout_seconds: int = 120


class TrainingConfig(BaseModel):
    # Architecture: "default" uses base_model directly; "cbam" loads yolov8n_cbam.yaml
    # and transfers pretrained backbone weights from base_model
    arch: str = "cbam"
    # Explicit architecture yaml path; when set it overrides _ARCH_YAML[arch].
    # Used by the experiment runner to point at configs/experiments/*.yaml variants.
    arch_yaml: Optional[str] = None
    base_model: str = "yolov8n.pt"
    epochs: int = 100
    batch: int = 16
    # 640 to match deployment (ai_service YOLO_INPUT_SIZE=640). Measured: dropping
    # to 320 loses ~16pts (recall 0.90->0.73, mAP50 0.94->0.79) — not worth it for
    # the trigger-based upload pipeline. Only lower this if Pi4 FPS becomes a hard limit.
    imgsz: int = 640
    patience: int = 20
    workers: int = 8
    device: str = ""
    # Dataset fraction used per epoch — 1.0 for full training, <1.0 for cheap
    # architecture screening (experiment runner sets ~0.25).
    fraction: float = 1.0
    # Pretrained weight transfer scope: "full" copies backbone+neck+box-head
    # positionally (safe for CBAM/ECA placement variants); "backbone" stops at the
    # base SPPF so topology-changing variants (e.g. P2 head) don't receive
    # mismatched neck weights — those layers stay randomly initialised.
    transfer_scope: str = "full"
    dataset: str = "./merged_dataset/data.yaml"
    runs_dir: str = "./runs"
    run_name: str = "passenger_yolov8"
    export: ExportConfig = ExportConfig()
    deploy: DeployConfig = DeployConfig()


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_dataset_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetConfig(**raw)


def load_training_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Env vars override YAML for credentials — keep secrets out of version control
    deploy = raw.setdefault("deploy", {})
    deploy["enabled"]    = os.getenv("CLOUDFLARE_DEPLOY_ENABLED", "").lower() == "true" or deploy.get("enabled", False)
    deploy["worker_url"] = os.getenv("CLOUDFLARE_WORKER_URL", deploy.get("worker_url", ""))
    deploy["api_key"]    = os.getenv("CLOUDFLARE_API_KEY",    deploy.get("api_key", ""))
    return TrainingConfig(**raw)
