import logging
from pathlib import Path

from ultralytics import YOLO

from ..core.config import TrainingConfig

logger = logging.getLogger(__name__)


class EvaluateService:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, weights: Path, data_yaml: Path) -> dict:
        """Run validation on the test split and return key metrics."""
        logger.info(f"Evaluating {weights.name} on test split ...")

        device = "cpu" if str(weights).endswith(".onnx") else (self.config.device or None)
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            device=device,
            verbose=False,
        )

        results = {
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50-95":  round(float(metrics.box.map),   4),
            "precision": round(float(metrics.box.mp),    4),
            "recall":    round(float(metrics.box.mr),    4),
        }

        logger.info("── Test-set results ──────────────────────────")
        for k, v in results.items():
            logger.info(f"  {k:<12}: {v:.4f}")
        logger.info("─────────────────────────────────────────────")
        return results
