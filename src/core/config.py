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
    base_model: str = "yolov8n.pt"
    epochs: int = 100
    batch: int = 16
    # 320 recommended for Pi4 (4× fewer ops vs 640, ~90% accuracy retained)
    imgsz: int = 320
    patience: int = 20
    workers: int = 8
    device: str = ""
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
