"""Unit tests for EvaluateService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import TrainingConfig
from src.services.evaluate_service import EvaluateService


def _make_config(batch: int = 4, imgsz: int = 640) -> TrainingConfig:
    return TrainingConfig(batch=batch, imgsz=imgsz)


def _make_mock_metrics(map50=0.85, map=0.62, mp=0.88, mr=0.79) -> MagicMock:
    metrics = MagicMock()
    metrics.box.map50 = map50
    metrics.box.map   = map
    metrics.box.mp    = mp
    metrics.box.mr    = mr
    return metrics


class TestEvaluateService:
    def test_calls_model_val_with_correct_params(self, tmp_path):
        weights = tmp_path / "best.onnx"
        weights.write_bytes(b"fake_onnx")
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\n")

        cfg = _make_config(batch=4, imgsz=640)
        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_metrics()

        with patch("src.services.evaluate_service.YOLO", return_value=mock_model):
            EvaluateService(cfg).run(weights, data_yaml)

        mock_model.val.assert_called_once_with(
            data=str(data_yaml),
            split="test",
            imgsz=640,
            batch=4,
            verbose=False,
        )

    def test_returns_metrics_dict(self, tmp_path):
        weights = tmp_path / "best.onnx"
        weights.write_bytes(b"fake_onnx")
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\n")

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_metrics(
            map50=0.85, map=0.62, mp=0.88, mr=0.79
        )

        with patch("src.services.evaluate_service.YOLO", return_value=mock_model):
            results = EvaluateService(cfg).run(weights, data_yaml)

        assert results == {
            "mAP50":     0.85,
            "mAP50-95":  0.62,
            "precision": 0.88,
            "recall":    0.79,
        }

    def test_metrics_rounded_to_4dp(self, tmp_path):
        weights = tmp_path / "best.onnx"
        weights.write_bytes(b"fake_onnx")
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\n")

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_metrics(
            map50=0.854321, map=0.621987, mp=0.883456, mr=0.791234
        )

        with patch("src.services.evaluate_service.YOLO", return_value=mock_model):
            results = EvaluateService(cfg).run(weights, data_yaml)

        for key, val in results.items():
            assert len(str(val).split(".")[-1]) <= 4, f"{key} has too many decimal places: {val}"

    def test_loads_correct_weights_file(self, tmp_path):
        weights = tmp_path / "best.onnx"
        weights.write_bytes(b"fake_onnx")
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\n")

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_metrics()

        with patch("src.services.evaluate_service.YOLO", return_value=mock_model) as mock_yolo:
            EvaluateService(cfg).run(weights, data_yaml)

        mock_yolo.assert_called_once_with(str(weights))
