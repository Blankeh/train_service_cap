"""Unit tests for TrainingService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import TrainingConfig
from src.services.training_service import TrainingService


def _make_config(runs_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        base_model="yolov8n.pt",
        epochs=1,
        batch=2,
        imgsz=320,
        patience=5,
        workers=1,
        device="cpu",
        runs_dir=str(runs_dir),
        run_name="test_run",
    )


class TestTrainingService:
    def test_raises_if_data_yaml_missing(self, tmp_path):
        cfg = _make_config(tmp_path / "runs")
        with pytest.raises(FileNotFoundError):
            TrainingService(cfg).run(tmp_path / "nonexistent.yaml")

    def test_calls_yolo_train_with_correct_params(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: [person]\n")
        cfg = _make_config(tmp_path / "runs")

        mock_model = MagicMock()
        with patch("src.services.training_service.YOLO", return_value=mock_model) as mock_yolo:
            # Pre-create best.pt so the return path check passes
            best = tmp_path / "runs" / "test_run" / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"fake_weights")

            result = TrainingService(cfg).run(data_yaml)

        mock_yolo.assert_called_once_with("yolov8n.pt")
        call_kwargs = mock_model.train.call_args.kwargs
        assert call_kwargs["data"] == str(data_yaml)
        assert call_kwargs["epochs"] == 1
        assert call_kwargs["batch"] == 2
        assert call_kwargs["imgsz"] == 320
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["project"] == str(tmp_path / "runs")
        assert call_kwargs["name"] == "test_run"

    def test_returns_best_pt_path(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: [person]\n")
        cfg = _make_config(tmp_path / "runs")

        mock_model = MagicMock()
        with patch("src.services.training_service.YOLO", return_value=mock_model):
            best = tmp_path / "runs" / "test_run" / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"fake_weights")

            result = TrainingService(cfg).run(data_yaml)

        assert result == best

    def test_device_none_when_empty_string(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: [person]\n")
        cfg = _make_config(tmp_path / "runs")
        cfg.device = ""

        mock_model = MagicMock()
        with patch("src.services.training_service.YOLO", return_value=mock_model):
            best = tmp_path / "runs" / "test_run" / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"fake_weights")
            TrainingService(cfg).run(data_yaml)

        assert mock_model.train.call_args.kwargs["device"] is None
