import logging
from pathlib import Path

from ultralytics import YOLO

from ..core.config import TrainingConfig

logger = logging.getLogger(__name__)


class ExportService:
    """
    Exports a trained .pt model to ONNX for Raspberry Pi 4 deployment.

    ONNX + ONNX Runtime gives the best CPU throughput on the Pi4's
    ARM Cortex-A72 while staying fully compatible with the Ultralytics
    inference pipeline on the device.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, weights: Path) -> Path:
        """Export weights to the configured format. Returns the exported file path."""
        cfg = self.config.export
        logger.info(f"Exporting {weights.name} -> {cfg.format.upper()}")
        logger.info(f"  simplify={cfg.simplify}  opset={cfg.opset}  int8={cfg.int8}")

        model = YOLO(str(weights))
        out_path = model.export(
            format=cfg.format,
            imgsz=self.config.imgsz,
            simplify=cfg.simplify,
            opset=cfg.opset,
            int8=cfg.int8,
            dynamic=False,
        )

        out = Path(out_path)
        size_mb = out.stat().st_size / 1_048_576
        logger.info(f"Exported: {out}  ({size_mb:.1f} MB)")
        return out
