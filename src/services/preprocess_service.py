import hashlib
import logging
import random
import shutil
from pathlib import Path

import yaml

from ..core.config import DatasetConfig

logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── Box-style conversion ratios ──────────────────────────────────────────────
# Derived from standard human body proportions (head ≈ 1/7 body height).
# Tuned slightly wider/taller than strict anatomy because annotations tend to
# include some neck/hair margin.
_BODY_TO_HEAD_H_RATIO = 0.20   # head height ≈ 20 % of full-body height
_BODY_TO_HEAD_W_RATIO = 0.50   # head width  ≈ 50 % of full-body width
_HEAD_TO_BODY_H_RATIO = 5.0    # inverse: body ≈ 5× head height
_HEAD_TO_BODY_W_RATIO = 2.0    # inverse: body ≈ 2× head width


class PreprocessService:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """Merge and deduplicate all configured datasets. Returns path to data.yaml."""
        out = Path(self.config.output.path)
        out.mkdir(parents=True, exist_ok=True)
        seen_hashes: set[str] = set()
        target_style = self.config.output.target_box_style

        for entry in self.config.datasets:
            ds_root = Path(entry.path)
            logger.info(f"Processing dataset: {entry.name}")
            logger.info(
                f"  box_style={entry.box_style}  target={target_style}"
                f"  {'(conversion needed)' if entry.box_style != target_style else '(no conversion)'}"
            )

            if entry.auto_split is None:
                self._process_with_existing_splits(
                    ds_root, entry, seen_hashes, out, target_style,
                )
            else:
                self._process_with_auto_split(
                    ds_root, entry, seen_hashes, out, target_style,
                )

        return self._write_data_yaml(out)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_with_existing_splits(
        self, ds_root: Path, entry, seen: set[str], out: Path, target_style: str,
    ) -> None:
        for split in entry.splits:
            img_dir = ds_root / split / "images"
            lbl_dir = ds_root / split / "labels"
            if not img_dir.exists():
                logger.warning(f"  [{split}] not found in {entry.name}, skipping")
                continue
            copied, skipped, stats = self._copy_split(
                img_dir, lbl_dir, entry.class_map,
                entry.box_style, target_style, seen,
                out / split / "images", out / split / "labels",
            )
            logger.info(f"  [{split}] copied={copied:,}  duplicates_removed={skipped}")
            self._log_conversion_stats(stats)

    def _process_with_auto_split(
        self, ds_root: Path, entry, seen: set[str], out: Path, target_style: str,
    ) -> None:
        img_dir = ds_root / "train" / "images"
        lbl_dir = ds_root / "train" / "labels"
        unique: list[tuple[Path, Path]] = []
        skipped = 0

        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in EXTENSIONS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            h = self._sha256(img)
            if h in seen:
                skipped += 1
                continue
            seen.add(h)
            unique.append((img, lbl))

        logger.info(f"  unique={len(unique):,}  duplicates_removed={skipped}")

        random.seed(self.config.output.seed)
        random.shuffle(unique)
        n = len(unique)
        n_train = int(n * entry.auto_split["train"])
        n_valid = int(n * entry.auto_split["valid"])
        buckets = {
            "train": unique[:n_train],
            "valid": unique[n_train : n_train + n_valid],
            "test":  unique[n_train + n_valid :],
        }
        for split, pairs in buckets.items():
            stats = self._copy_pairs(
                pairs, entry.class_map,
                entry.box_style, target_style,
                out / split / "images", out / split / "labels",
            )
            logger.info(f"  [{split}] added {len(pairs)}")
            self._log_conversion_stats(stats)

    def _copy_split(
        self,
        img_dir: Path,
        lbl_dir: Path,
        class_map: dict[int, int],
        source_style: str,
        target_style: str,
        seen: set[str],
        out_img: Path,
        out_lbl: Path,
    ) -> tuple[int, int, dict]:
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        copied = skipped = 0
        stats = {"polygons_converted": 0, "boxes_restyled": 0, "labels_written": 0}
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in EXTENSIONS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            h = self._sha256(img)
            if h in seen:
                skipped += 1
                continue
            seen.add(h)
            shutil.copy2(img, out_img / img.name)
            file_stats = self._remap_label(
                lbl, out_lbl / lbl.name,
                class_map, source_style, target_style,
            )
            for k in stats:
                stats[k] += file_stats.get(k, 0)
            copied += 1
        return copied, skipped, stats

    def _copy_pairs(
        self,
        pairs: list[tuple[Path, Path]],
        class_map: dict[int, int],
        source_style: str,
        target_style: str,
        out_img: Path,
        out_lbl: Path,
    ) -> dict:
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        stats = {"polygons_converted": 0, "boxes_restyled": 0, "labels_written": 0}
        for img, lbl in pairs:
            shutil.copy2(img, out_img / img.name)
            file_stats = self._remap_label(
                lbl, out_lbl / lbl.name,
                class_map, source_style, target_style,
            )
            for k in stats:
                stats[k] += file_stats.get(k, 0)
        return stats

    def _write_data_yaml(self, out: Path) -> Path:
        classes = self.config.output.classes
        data = {
            "path":  str(out.resolve()),
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    len(classes),
            "names": classes,
        }
        yaml_path = out / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"data.yaml written to {yaml_path}")
        return yaml_path

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    # ── Label conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _remap_label(
        src: Path,
        dst: Path,
        class_map: dict[int, int],
        source_style: str,
        target_style: str,
    ) -> dict:
        """
        Read a YOLO label file, apply class remapping, convert polygons to
        bounding boxes if needed, and normalize box style (head ↔ body).

        Returns per-file conversion stats.
        """
        stats = {"polygons_converted": 0, "boxes_restyled": 0, "labels_written": 0}
        text = src.read_text(encoding="utf-8").strip()
        if not text:
            return stats

        lines = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = class_map.get(int(parts[0]), 0)
            values = [float(v) for v in parts[1:]]

            # ── Step 1: polygon → bbox ────────────────────────────────────
            if len(values) > 4:
                # Segmentation polygon: x1 y1 x2 y2 x3 y3 ...
                xs = values[0::2]
                ys = values[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w  = x_max - x_min
                h  = y_max - y_min
                values = [cx, cy, w, h]
                stats["polygons_converted"] += 1

            cx, cy, w, h = values

            # ── Step 2: box-style normalization ───────────────────────────
            if source_style != target_style:
                cx, cy, w, h = PreprocessService._convert_box(
                    cx, cy, w, h, source_style, target_style,
                )
                stats["boxes_restyled"] += 1

            lines.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )

        if lines:
            dst.write_text("\n".join(lines), encoding="utf-8")
            stats["labels_written"] = 1
        return stats

    @staticmethod
    def _convert_box(
        cx: float, cy: float, w: float, h: float,
        source: str, target: str,
    ) -> tuple[float, float, float, float]:
        """
        Geometrically convert between head and body box styles.

        body → head:  Take the top ~20 % height, ~50 % width of the body box.
        head → body:  Expand downward by inverse ratios.

        All values are clamped to [0, 1] (normalised image coordinates).
        """
        if source == "body" and target == "head":
            new_h  = h * _BODY_TO_HEAD_H_RATIO
            new_w  = w * _BODY_TO_HEAD_W_RATIO
            # Anchor to top of body box: body_top = cy - h/2
            new_cy = (cy - h / 2) + new_h / 2
            new_cx = cx  # centred horizontally
        elif source == "head" and target == "body":
            new_h  = h * _HEAD_TO_BODY_H_RATIO
            new_w  = w * _HEAD_TO_BODY_W_RATIO
            # Expand downward from head position
            new_cy = cy + new_h / 2 - h / 2
            new_cx = cx
        else:
            return cx, cy, w, h

        # Clamp to valid normalised range
        new_cx = max(0.0, min(1.0, new_cx))
        new_cy = max(0.0, min(1.0, new_cy))
        new_w  = max(0.001, min(1.0, new_w))
        new_h  = max(0.001, min(1.0, new_h))
        return new_cx, new_cy, new_w, new_h

    @staticmethod
    def _log_conversion_stats(stats: dict) -> None:
        parts = []
        if stats.get("polygons_converted"):
            parts.append(f"polygons→bbox={stats['polygons_converted']}")
        if stats.get("boxes_restyled"):
            parts.append(f"restyled={stats['boxes_restyled']}")
        if parts:
            logger.info(f"    conversions: {', '.join(parts)}")
