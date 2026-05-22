import logging
from pathlib import Path

from ultralytics import YOLO

from ..core.config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, data_yaml: Path) -> Path:
        """Train YOLOv8 on the merged dataset. Returns path to best.pt."""
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        logger.info(f"Base model : {self.config.base_model}")
        logger.info(f"Epochs     : {self.config.epochs}")
        logger.info(f"Batch      : {self.config.batch}")
        logger.info(f"Image size : {self.config.imgsz}")
        logger.info(f"Device     : {self.config.device or 'auto'}")

        model = YOLO(self.config.base_model)
        model.train(
            data=str(data_yaml),
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            patience=self.config.patience,
            workers=self.config.workers,
            device=self.config.device or None,
            project=self.config.runs_dir,
            name=self.config.run_name,
            pretrained=True,
            save=True,
            plots=True,
            verbose=False,
        )

        best = Path(model.trainer.best)
        logger.info(f"Training complete — best weights: {best}")
        return best
