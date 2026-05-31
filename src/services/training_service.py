import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from ..core.config import TrainingConfig
from ..core.custom_modules import register_custom_modules

logger = logging.getLogger(__name__)

_ARCH_YAML = {
    "cbam": "configs/yolov8n_cbam.yaml",
}


def _transfer_backbone_weights(custom: YOLO, pretrained_tag: str) -> None:
    """
    Copy matching weights from a standard pretrained model into the custom model.

    Uses shape-based matching (strict=False equivalent) so that identical
    backbone layers get the pretrained weights while new attention layers
    (CBAM) keep their random initialisation.
    """
    logger.info(f"Transferring backbone weights from {pretrained_tag} ...")
    base = YOLO(pretrained_tag)

    custom_sd = custom.model.state_dict()
    base_sd = base.model.state_dict()

    matched = 0
    for key, tensor in custom_sd.items():
        if key in base_sd and base_sd[key].shape == tensor.shape:
            custom_sd[key] = base_sd[key]
            matched += 1

    custom.model.load_state_dict(custom_sd)
    total = len(custom_sd)
    logger.info(f"  Transferred {matched}/{total} tensors "
                f"({matched / total * 100:.1f}% — CBAM layers randomly initialised)")


class TrainingService:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, data_yaml: Path) -> Path:
        """Train on the merged dataset. Returns path to best.pt."""
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        cfg = self.config

        # Register CBAM / ECA into Ultralytics so parse_model can find them
        register_custom_modules()

        if cfg.arch == "default":
            model = YOLO(cfg.base_model)
            logger.info(f"Architecture   : default ({cfg.base_model})")
        else:
            yaml_path = Path(_ARCH_YAML.get(cfg.arch, _ARCH_YAML["cbam"]))
            if not yaml_path.exists():
                raise FileNotFoundError(f"Model YAML not found: {yaml_path}")
            model = YOLO(str(yaml_path))
            logger.info(f"Architecture   : {cfg.arch} ({yaml_path.name})")
            _transfer_backbone_weights(model, cfg.base_model)

        logger.info(f"Base model     : {cfg.base_model}")
        logger.info(f"Epochs         : {cfg.epochs}")
        logger.info(f"Batch          : {cfg.batch}")
        logger.info(f"Image size     : {cfg.imgsz}")
        logger.info(f"Device         : {cfg.device or 'auto'}")

        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            patience=cfg.patience,
            workers=cfg.workers,
            device=cfg.device or None,
            project=cfg.runs_dir,
            name=cfg.run_name,
            pretrained=False,       # backbone transfer already done above
            save=True,
            plots=True,
            verbose=False,
            # ── LR: slightly lower so pretrained backbone doesn't shift too fast
            lr0=0.005,
            lrf=0.01,
            warmup_epochs=5,        # let random CBAM weights stabilise before full LR
            # ── Augmentation: small dataset needs heavy aug to prevent CBAM overfitting
            copy_paste=0.3,         # paste objects across images — good for crowded scenes
            mixup=0.15,
            degrees=10.0,
            scale=0.6,
            close_mosaic=20,        # disable mosaic last 20 epochs for cleaner box regression
            # ── Regularisation
            weight_decay=0.0005,
            dropout=0.0,
        )

        best = Path(model.trainer.best)
        logger.info(f"Training complete — best weights: {best}")
        return best
