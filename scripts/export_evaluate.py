import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_training_config
from src.services.export_service import ExportService
from src.services.evaluate_service import EvaluateService

cfg = load_training_config(Path("configs/training.yaml"))
best_pt = Path("runs/detect/runs/passenger_yolov8-2/weights/best.pt")
best_onnx = ExportService(cfg).run(best_pt)
cfg.device = "cpu"
EvaluateService(cfg).run(best_onnx, Path("merged_dataset/data.yaml"))
