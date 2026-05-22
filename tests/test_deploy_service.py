"""Unit tests for DeployService."""

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import DeployConfig, TrainingConfig
from src.services.deploy_service import DeployService


def _make_config(worker_url: str = "https://worker.example.com", api_key: str = "key123") -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.deploy = DeployConfig(
        enabled=True,
        worker_url=worker_url,
        api_key=api_key,
        timeout_seconds=30,
    )
    return cfg


def _make_model(tmp_path: Path, content: bytes = b"fake_onnx_model") -> Path:
    model = tmp_path / "best.onnx"
    model.write_bytes(content)
    return model


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestDeployService:
    def test_posts_to_upload_endpoint(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "v1", "sha256": _sha256(b"fake_onnx_model")}

        with patch("src.services.deploy_service.httpx.post", return_value=mock_response) as mock_post:
            DeployService(cfg).run(model)

        call_url = mock_post.call_args.args[0]
        assert call_url == "https://worker.example.com/api/model/upload"

    def test_sends_auth_header(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config(api_key="secret-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "v1", "sha256": _sha256(b"fake_onnx_model")}

        with patch("src.services.deploy_service.httpx.post", return_value=mock_response) as mock_post:
            DeployService(cfg).run(model)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-key"

    def test_exits_on_http_error(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("src.services.deploy_service.httpx.post", return_value=mock_response):
            with pytest.raises(SystemExit):
                DeployService(cfg).run(model)

    def test_exits_if_model_file_missing(self, tmp_path):
        cfg = _make_config()
        with pytest.raises(SystemExit):
            DeployService(cfg).run(tmp_path / "nonexistent.onnx")

    def test_exits_if_worker_url_empty(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config(worker_url="")
        with pytest.raises(SystemExit):
            DeployService(cfg).run(model)

    def test_exits_if_api_key_empty(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config(api_key="")
        with pytest.raises(SystemExit):
            DeployService(cfg).run(model)

    def test_strips_trailing_slash_from_url(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config(worker_url="https://worker.example.com/")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "v1", "sha256": _sha256(b"fake_onnx_model")}

        with patch("src.services.deploy_service.httpx.post", return_value=mock_response) as mock_post:
            DeployService(cfg).run(model)

        call_url = mock_post.call_args.args[0]
        assert not call_url.startswith("https://worker.example.com//")
        assert call_url == "https://worker.example.com/api/model/upload"

    def test_uses_configured_timeout(self, tmp_path):
        model = _make_model(tmp_path)
        cfg = _make_config()
        cfg.deploy.timeout_seconds = 60

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "v1", "sha256": _sha256(b"fake_onnx_model")}

        with patch("src.services.deploy_service.httpx.post", return_value=mock_response) as mock_post:
            DeployService(cfg).run(model)

        assert mock_post.call_args.kwargs["timeout"] == 60
