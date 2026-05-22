import hashlib
import logging
import sys
from pathlib import Path

import httpx

from ..core.config import TrainingConfig

logger = logging.getLogger(__name__)


class DeployService:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, model_path: Path) -> None:
        cfg = self.config.deploy

        if not cfg.worker_url:
            logger.error("CLOUDFLARE_WORKER_URL is not set — cannot deploy")
            sys.exit(1)
        if not cfg.api_key:
            logger.error("CLOUDFLARE_API_KEY is not set — cannot deploy")
            sys.exit(1)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        sha256 = self._sha256(model_path)
        size_mb = model_path.stat().st_size / 1_048_576
        url = cfg.worker_url.rstrip("/") + "/api/model/upload"

        logger.info(f"Uploading {model_path.name}  ({size_mb:.1f} MB)  ->  {url}")
        logger.info(f"  SHA-256: {sha256[:16]}...")

        with open(model_path, "rb") as f:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                files={"file": (model_path.name, f, "application/octet-stream")},
                timeout=cfg.timeout_seconds,
            )

        if response.status_code == 200:
            body = response.json()
            logger.info(
                f"Uploaded — version={body.get('version')}  "
                f"sha256={str(body.get('sha256', ''))[:16]}..."
            )
            logger.info("Pi4 will pick up the new model on its next poll cycle")
        else:
            logger.error(f"Upload failed  HTTP {response.status_code}: {response.text}")
            sys.exit(1)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
