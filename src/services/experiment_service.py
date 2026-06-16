"""
Architecture-ablation runner: CBAM-placement variants + a P2 small-object head.

Strategy (see plan): a cheap SCREENING pass trains every variant from scratch
(pretrained transfer, NOT fine-tuning the converged model) on a fraction of the
data for few epochs at the deployment resolution, ranks them, then the winner is
FULL-trained separately.

Each variant is evaluated on two sets:
  - merged_dataset val  → the real distribution (ranking metric)
  - passenger_eval      → the clean apples-to-apples set (continuity with run-7)

Reuses TrainingService (build + pretrained transfer + train) and EvaluateService
(metrics) rather than duplicating that logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from ..core.config import TrainingConfig
from ..core.custom_modules import register_custom_modules
from .evaluate_service import EvaluateService
from .training_service import TrainingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    arch_yaml: str
    transfer_scope: str = "full"


# Baseline must be re-screened under the SAME cheap budget — run-7's full-data
# 300-epoch numbers are not comparable to a 40-epoch / 25%-data screen.
VARIANTS: list[Variant] = [
    Variant("baseline",      "configs/yolov8n_cbam.yaml"),
    Variant("p3only",        "configs/experiments/yolov8n_cbam_p3only.yaml"),
    Variant("backbone",      "configs/experiments/yolov8n_cbam_backbone.yaml"),
    Variant("backbone_neck", "configs/experiments/yolov8n_cbam_backbone_neck.yaml"),
    Variant("p2head",        "configs/experiments/yolov8n_cbam_p2head.yaml", "backbone"),
]

# Cheap screening budget.
SCREEN_OVERRIDES = dict(imgsz=640, batch=32, epochs=40, patience=10, fraction=0.25)
# Full training budget for the winner.
FULL_OVERRIDES = dict(imgsz=640, batch=32, epochs=300, patience=30, fraction=1.0)

PASSENGER_EVAL = Path("configs/experiments/passenger_eval.yaml")
RESULTS_DIR = Path("runs/experiments")


def _model_stats(weights: Path) -> tuple[int, float]:
    """Return (params, GFLOPs) for a trained checkpoint."""
    register_custom_modules()
    m = YOLO(str(weights))
    info = m.info(verbose=False)  # (layers, params, gradients, gflops)
    params = int(info[1]) if info and len(info) > 1 else sum(p.numel() for p in m.model.parameters())
    gflops = float(info[3]) if info and len(info) > 3 else float("nan")
    return params, gflops


class ExperimentService:
    def __init__(self, base_cfg: TrainingConfig) -> None:
        self.base_cfg = base_cfg
        self.data_yaml = Path(base_cfg.dataset)

    def _cfg_for(self, v: Variant, overrides: dict, prefix: str) -> TrainingConfig:
        return self.base_cfg.model_copy(update={
            "arch": "cbam",
            "arch_yaml": v.arch_yaml,
            "transfer_scope": v.transfer_scope,
            "runs_dir": str(RESULTS_DIR),
            "run_name": f"{prefix}_{v.name}",
            **overrides,
        })

    def _train_and_eval(self, v: Variant, overrides: dict, prefix: str) -> dict:
        cfg = self._cfg_for(v, overrides, prefix)
        logger.info("══ Variant %s (%s, scope=%s) ════════════════",
                    v.name, Path(v.arch_yaml).name, v.transfer_scope)
        best = TrainingService(cfg).run(self.data_yaml)

        evaluator = EvaluateService(cfg)
        merged = evaluator.run(best, self.data_yaml, split="val")
        passenger = evaluator.run(best, PASSENGER_EVAL, split="val")
        params, gflops = _model_stats(best)

        return {
            "variant": v.name,
            "weights": str(best),
            "params": params,
            "gflops": round(gflops, 2),
            "merged_mAP50": merged["mAP50"],
            "merged_recall": merged["recall"],
            "pass_mAP50": passenger["mAP50"],
            "pass_recall": passenger["recall"],
        }

    def screen(self, variants: list[Variant] | None = None) -> list[dict]:
        """Cheap screening pass over all variants; writes a ranked RESULTS.md."""
        variants = variants or VARIANTS
        rows = [self._train_and_eval(v, SCREEN_OVERRIDES, "screen") for v in variants]
        rows.sort(key=lambda r: (r["merged_recall"], r["merged_mAP50"]), reverse=True)
        self._write_results(rows, "Screening (640, 25% data, 40 epochs)")
        logger.info("Screening winner (by merged-val recall): %s", rows[0]["variant"])
        return rows

    def full(self, variant_name: str) -> dict:
        """Full retrain + dual eval of a single chosen variant."""
        v = next((x for x in VARIANTS if x.name == variant_name), None)
        if v is None:
            raise ValueError(f"Unknown variant {variant_name!r}; choose from "
                             f"{[x.name for x in VARIANTS]}")
        row = self._train_and_eval(v, FULL_OVERRIDES, "full")
        self._write_results([row], f"Full train — {variant_name}")
        # Honest gate: compare to run-7.
        logger.info("run-7 baseline for reference: passenger recall 0.895 / "
                    "merged recall 0.656. This variant: passenger %.3f / merged %.3f",
                    row["pass_recall"], row["merged_recall"])
        return row

    def _write_results(self, rows: list[dict], title: str) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "RESULTS.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = (
            "| variant | params | GFLOPs | merged mAP50 | merged recall | "
            "pass mAP50 | pass recall |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        body = "".join(
            f"| {r['variant']} | {r['params']:,} | {r['gflops']} | "
            f"{r['merged_mAP50']} | {r['merged_recall']} | "
            f"{r['pass_mAP50']} | {r['pass_recall']} |\n"
            for r in rows
        )
        block = f"\n## {title} — {ts}\n\nReference: run-7 = merged recall 0.656 / passenger recall 0.895\n\n{header}{body}"
        with open(out, "a", encoding="utf-8") as f:
            f.write(block)
        logger.info("Results appended to %s", out)
