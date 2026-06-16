import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from ..core.config import TrainingConfig
from ..core.custom_modules import CBAM, ECA, register_custom_modules

logger = logging.getLogger(__name__)

_ARCH_YAML = {
    "cbam": "configs/yolov8n_cbam.yaml",
}


# Base YOLOv8n backbone is layers 0..9 (Conv stem … SPPF). The neck/head start
# at index 10. "backbone" transfer scope stops here.
_BASE_BACKBONE_LAST = 9


def _transfer_pretrained_weights(
    custom: YOLO, pretrained_tag: str, scope: str = "full"
) -> None:
    """
    Copy pretrained weights into the custom model — backbone, neck AND head.

    The CBAM yaml is the standard YOLOv8n graph with attention blocks inserted
    after each detection-scale C2f. Those insertions shift every later layer's
    index, so a plain key match only recovers the layers before the first CBAM
    (~58%), leaving the bottom-up neck and the entire Detect head randomly
    initialised — which cripples box/class prediction.

    Instead we align the two module lists positionally: the custom model is the
    base model with CBAM/ECA layers inserted, so we walk both and skip the
    custom-only attention layers, mapping every standard layer to its base
    counterpart. Shape-checked transfer then fills backbone + neck + Detect box
    head with pretrained weights; only the attention blocks and the class head
    (reinitialised whenever nc differs from COCO's 80) start random — exactly
    like normal YOLO fine-tuning.

    scope:
        "full"     — map every standard layer (placement variants whose neck is
                     still the base neck with attention inserted).
        "backbone" — only map standard layers up to the base SPPF (idx 9). Use
                     for topology-changing variants (e.g. a P2 head) whose neck
                     diverges from base: positional alignment past the backbone
                     would copy mismatched weights, so leave the neck/head random.
    """
    logger.info(
        f"Transferring pretrained weights from {pretrained_tag} (scope={scope}) ..."
    )
    base = YOLO(pretrained_tag)

    # Positionally align custom<->base, skipping custom-only attention layers.
    idx_map: dict[int, int] = {}
    b = 0
    base_layers = base.model.model
    for c, layer in enumerate(custom.model.model):
        if isinstance(layer, (CBAM, ECA)):
            continue  # custom-only — keep random init, don't consume a base layer
        if scope == "backbone" and b > _BASE_BACKBONE_LAST:
            break     # neck/head diverges — stop, leave the rest randomly init
        if b < len(base_layers):
            idx_map[c] = b
            b += 1

    custom_sd = custom.model.state_dict()
    base_sd = base.model.state_dict()

    matched = 0
    for key, tensor in custom_sd.items():
        parts = key.split(".")  # "model.<idx>.<rest>"
        if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit():
            c_idx = int(parts[1])
            if c_idx in idx_map:
                base_key = ".".join(["model", str(idx_map[c_idx]), *parts[2:]])
                if base_key in base_sd and base_sd[base_key].shape == tensor.shape:
                    custom_sd[key] = base_sd[base_key]
                    matched += 1

    custom.model.load_state_dict(custom_sd)
    total = len(custom_sd)
    random_note = ("attention + class head random" if scope == "full"
                   else "entire neck/head random (backbone-only transfer)")
    logger.info(f"  Transferred {matched}/{total} tensors "
                f"({matched / total * 100:.1f}% — {random_note})")


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
            # Explicit arch_yaml (experiment variants) overrides the _ARCH_YAML map.
            yaml_path = (
                Path(cfg.arch_yaml) if cfg.arch_yaml
                else Path(_ARCH_YAML.get(cfg.arch, _ARCH_YAML["cbam"]))
            )
            if not yaml_path.exists():
                raise FileNotFoundError(f"Model YAML not found: {yaml_path}")
            model = YOLO(str(yaml_path))
            logger.info(f"Architecture   : {cfg.arch} ({yaml_path.name})")
            _transfer_pretrained_weights(model, cfg.base_model, cfg.transfer_scope)

        logger.info(f"Base model     : {cfg.base_model}")
        logger.info(f"Epochs         : {cfg.epochs}")
        logger.info(f"Batch          : {cfg.batch}")
        logger.info(f"Image size     : {cfg.imgsz}")
        logger.info(f"Data fraction  : {cfg.fraction}")
        logger.info(f"Device         : {cfg.device or 'auto'}")

        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            patience=cfg.patience,
            fraction=cfg.fraction,   # <1.0 for cheap architecture screening
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
            # ── Augmentation: merged_dataset is now large (~40k imgs), so the heavy
            # synthetic aug that guarded against overfitting on the old tiny set is
            # no longer needed and was capping recall. Keep mild geometric aug only.
            copy_paste=0.0,         # was 0.3 — unnecessary at this dataset size
            mixup=0.0,              # was 0.15 — blends hurt single-class person recall
            erasing=0.2,            # was default 0.4 — less aggressive occlusion
            degrees=5.0,            # was 10.0
            scale=0.5,              # was 0.6
            mosaic=0.5,             # was default 1.0
            close_mosaic=10,        # disable mosaic last 10 epochs for cleaner box regression
            # ── Regularisation
            weight_decay=0.0005,
            dropout=0.0,
        )

        best = Path(model.trainer.best)
        logger.info(f"Training complete — best weights: {best}")
        return best
