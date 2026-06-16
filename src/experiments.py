import argparse
import logging
from pathlib import Path

from .core.config import load_training_config
from .services.experiment_service import VARIANTS, ExperimentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Architecture ablation — CBAM placement + P2 head experiments")
    parser.add_argument("--training", default="configs/training.yaml",
                        help="Base training config (budget/LR/aug inherited)")
    parser.add_argument("--screen", action="store_true",
                        help="Cheap screening pass over all variants, ranked")
    parser.add_argument("--full", metavar="VARIANT", default=None,
                        help=f"Full-train one variant: {[v.name for v in VARIANTS]}")
    args = parser.parse_args()

    cfg = load_training_config(Path(args.training))
    svc = ExperimentService(cfg)

    if args.full:
        svc.full(args.full)
    elif args.screen:
        svc.screen()
    else:
        parser.error("pass --screen (rank all variants) or --full <variant>")


if __name__ == "__main__":
    main()
