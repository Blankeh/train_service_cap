import logging
from pathlib import Path

from ultralytics import YOLO

from ..core.config import TrainingConfig
from ..core.custom_modules import register_custom_modules

logger = logging.getLogger(__name__)


class ExportService:
    """
    Exports a trained .pt to deployment formats.

    Always produces ONNX (single file — used by deploy pipeline + evaluation).
    If export.format == 'ncnn', additionally exports NCNN for local Pi4 use.
    NCNN is a folder (.param + .bin) and cannot be uploaded as a single file,
    so it is not used in the OTA deploy path.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, weights: Path) -> Path:
        """
        Export weights. Always returns the ONNX path (used by eval + deploy).
        NCNN folder is logged separately for manual Pi4 copy.
        """
        register_custom_modules()

        cfg = self.config.export

        # ── ONNX (always — single file needed for eval and OTA deploy) ────────
        logger.info(f"Exporting {weights.name} → ONNX")
        onnx_path = self._export_onnx(weights)

        # ── NCNN (additionally, when configured — for local Pi4 performance) ──
        if cfg.format.lower() == "ncnn":
            try:
                ncnn_path = self._export_ncnn(weights)
                logger.info(f"NCNN model ready for Pi4 (copy folder to device): {ncnn_path}")
            except Exception as exc:
                logger.warning(f"NCNN export failed ({exc}) — ONNX will be used")

        elif cfg.format.lower() == "openvino":
            try:
                ov_path = self._export_openvino(weights)
                logger.info(f"OpenVINO model: {ov_path}")
            except Exception as exc:
                logger.warning(f"OpenVINO export failed ({exc}) — ONNX will be used")

        return onnx_path

    # ── Format helpers ────────────────────────────────────────────────────────

    def _export_onnx(self, weights: Path) -> Path:
        cfg = self.config.export
        model = YOLO(str(weights))
        out_path = model.export(
            format="onnx",
            imgsz=self.config.imgsz,
            simplify=cfg.simplify,
            opset=cfg.opset,
            int8=cfg.int8,
            dynamic=False,
        )
        out = Path(out_path)
        logger.info(f"ONNX: {out}  ({out.stat().st_size / 1_048_576:.1f} MB)")
        return out

    def _export_ncnn(self, weights: Path) -> Path:
        model = YOLO(str(weights))
        out_path = model.export(
            format="ncnn",
            imgsz=self.config.imgsz,
            int8=self.config.export.int8,
            half=False,
        )
        out = Path(out_path)
        logger.info(f"NCNN: {out}  ({out.stat().st_size / 1_048_576:.1f} MB)")
        return out

    def _export_openvino(self, weights: Path) -> Path:
        model = YOLO(str(weights))
        out_path = model.export(
            format="openvino",
            imgsz=self.config.imgsz,
            int8=self.config.export.int8,
            half=False,
        )
        out = Path(out_path)
        logger.info(f"OpenVINO: {out}  ({out.stat().st_size / 1_048_576:.1f} MB)")
        return out
