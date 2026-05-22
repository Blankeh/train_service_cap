import argparse
import logging
from pathlib import Path

from .core.config import load_dataset_config, load_training_config
from .services.preprocess_service import PreprocessService
from .services.training_service import TrainingService
from .services.export_service import ExportService
from .services.evaluate_service import EvaluateService
from .services.deploy_service import DeployService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Passenger counting — model training pipeline")
    parser.add_argument("--dataset",         default="configs/dataset.yaml",  help="Dataset config")
    parser.add_argument("--training",        default="configs/training.yaml", help="Training config")
    parser.add_argument("--skip-preprocess", action="store_true",             help="Skip if merged_dataset exists")
    parser.add_argument("--deploy",          action="store_true",             help="Push model to Pi4 after training")
    args = parser.parse_args()

    dataset_cfg  = load_dataset_config(Path(args.dataset))
    training_cfg = load_training_config(Path(args.training))

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    data_yaml = Path(training_cfg.dataset)
    if not args.skip_preprocess:
        logger.info("══ Stage 1/4  Preprocess ════════════════════")
        data_yaml = PreprocessService(dataset_cfg).run()
    else:
        logger.info(f"══ Stage 1/4  Preprocess  [skipped — {data_yaml}]")
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # ── 2. Train ───────────────────────────────────────────────────────────────
    logger.info("══ Stage 2/4  Train ═════════════════════════")
    best_pt = TrainingService(training_cfg).run(data_yaml)

    # ── 3. Export ─────────────────────────────────────────────────────────────
    logger.info("══ Stage 3/4  Export ════════════════════════")
    best_onnx = ExportService(training_cfg).run(best_pt)

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    logger.info("══ Stage 4/4  Evaluate ══════════════════════")
    EvaluateService(training_cfg).run(best_onnx, data_yaml)

    # ── 5. Deploy (optional) ──────────────────────────────────────────────────
    if args.deploy or training_cfg.deploy.enabled:
        logger.info("══ Stage 5/5  Deploy ════════════════════════")
        DeployService(training_cfg).run(best_onnx)
    else:
        logger.info("Deploy skipped (set CLOUDFLARE_DEPLOY_ENABLED=true or pass --deploy)")

    logger.info("Pipeline complete")
    logger.info(f"  PT   : {best_pt}")
    logger.info(f"  ONNX : {best_onnx}")


if __name__ == "__main__":
    main()
