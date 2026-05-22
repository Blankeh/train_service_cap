"""Unit tests for ExportService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import ExportConfig, TrainingConfig
from src.services.export_service import ExportService


def _make_config(imgsz: int = 640) -> TrainingConfig:
    return TrainingConfig(
        imgsz=imgsz,
        export=ExportConfig(format="onnx", simplify=True, opset=17, int8=False),
    )


class TestExportService:
    def test_calls_yolo_export_with_correct_params(self, tmp_path):
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake_weights")

        onnx_out = tmp_path / "best.onnx"
        onnx_out.write_bytes(b"fake_onnx")

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_out)

        cfg = _make_config(imgsz=640)
        with patch("src.services.export_service.YOLO", return_value=mock_model):
            result = ExportService(cfg).run(weights)

        mock_model.export.assert_called_once_with(
            format="onnx",
            imgsz=640,
            simplify=True,
            opset=17,
            int8=False,
            dynamic=False,
        )

    def test_returns_exported_path(self, tmp_path):
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake_weights")

        onnx_out = tmp_path / "best.onnx"
        onnx_out.write_bytes(b"fake_onnx")

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_out)

        cfg = _make_config()
        with patch("src.services.export_service.YOLO", return_value=mock_model):
            result = ExportService(cfg).run(weights)

        assert result == onnx_out

    def test_loads_correct_weights_file(self, tmp_path):
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake_weights")

        onnx_out = tmp_path / "best.onnx"
        onnx_out.write_bytes(b"fake_onnx")

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_out)

        cfg = _make_config()
        with patch("src.services.export_service.YOLO", return_value=mock_model) as mock_yolo:
            ExportService(cfg).run(weights)

        mock_yolo.assert_called_once_with(str(weights))

    def test_int8_flag_forwarded(self, tmp_path):
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake_weights")

        onnx_out = tmp_path / "best.onnx"
        onnx_out.write_bytes(b"fake_onnx")

        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_out)

        cfg = _make_config()
        cfg.export.int8 = True
        with patch("src.services.export_service.YOLO", return_value=mock_model):
            ExportService(cfg).run(weights)

        assert mock_model.export.call_args.kwargs["int8"] is True
