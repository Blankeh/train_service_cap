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
    splits: list[str]
    auto_split: Optional[dict[str, float]] = None


class DatasetOutputConfig(BaseModel):
    path: str = "./merged_dataset"
    classes: list[str] = ["person"]
    seed: int = 42


class DatasetConfig(BaseModel):
    datasets: list[DatasetEntry]
    output: DatasetOutputConfig


# ── Training config (configs/training.yaml) ───────────────────────────────────

class ExportConfig(BaseModel):
    format: str = "onnx"
    simplify: bool = True
    opset: int = 17
    int8: bool = False


class DeployConfig(BaseModel):
    enabled: bool = False      # opt-in via CLOUDFLARE_DEPLOY_ENABLED=true
    worker_url: str = ""       # e.g. https://your-worker.workers.dev
    api_key: str = ""
    timeout_seconds: int = 120


class TrainingConfig(BaseModel):
    base_model: str = "yolov8n.pt"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
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
