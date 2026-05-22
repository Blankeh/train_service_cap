import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import load_training_config
from src.services.deploy_service import DeployService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True,                   help="Path to .onnx or .pt")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config")
    args = parser.parse_args()

    cfg = load_training_config(Path(args.config))
    DeployService(cfg).run(Path(args.model))


if __name__ == "__main__":
    main()
