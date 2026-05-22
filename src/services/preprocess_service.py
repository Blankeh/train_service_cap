import hashlib
import logging
import random
import shutil
from pathlib import Path

import yaml

from ..core.config import DatasetConfig

logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class PreprocessService:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """Merge and deduplicate all configured datasets. Returns path to data.yaml."""
        out = Path(self.config.output.path)
        out.mkdir(parents=True, exist_ok=True)
        seen_hashes: set[str] = set()

        for entry in self.config.datasets:
            ds_root = Path(entry.path)
            logger.info(f"Processing dataset: {entry.name}")

            if entry.auto_split is None:
                self._process_with_existing_splits(ds_root, entry, seen_hashes, out)
            else:
                self._process_with_auto_split(ds_root, entry, seen_hashes, out)

        return self._write_data_yaml(out)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_with_existing_splits(
        self, ds_root: Path, entry, seen: set[str], out: Path
    ) -> None:
        for split in entry.splits:
            img_dir = ds_root / split / "images"
            lbl_dir = ds_root / split / "labels"
            if not img_dir.exists():
                logger.warning(f"  [{split}] not found in {entry.name}, skipping")
                continue
            copied, skipped = self._copy_split(
                img_dir, lbl_dir, entry.class_map, seen,
                out / split / "images", out / split / "labels",
            )
            logger.info(f"  [{split}] copied={copied:,}  duplicates_removed={skipped}")

    def _process_with_auto_split(
        self, ds_root: Path, entry, seen: set[str], out: Path
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
            self._copy_pairs(pairs, entry.class_map, out / split / "images", out / split / "labels")
            logger.info(f"  [{split}] added {len(pairs)}")

    def _copy_split(
        self,
        img_dir: Path,
        lbl_dir: Path,
        class_map: dict[int, int],
        seen: set[str],
        out_img: Path,
        out_lbl: Path,
    ) -> tuple[int, int]:
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        copied = skipped = 0
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
            self._remap_label(lbl, out_lbl / lbl.name, class_map)
            copied += 1
        return copied, skipped

    def _copy_pairs(
        self,
        pairs: list[tuple[Path, Path]],
        class_map: dict[int, int],
        out_img: Path,
        out_lbl: Path,
    ) -> None:
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy2(img, out_img / img.name)
            self._remap_label(lbl, out_lbl / lbl.name, class_map)

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

    @staticmethod
    def _remap_label(src: Path, dst: Path, class_map: dict[int, int]) -> None:
        text = src.read_text(encoding="utf-8").strip()
        if not text:
            return
        lines = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            lines.append(f"{class_map.get(int(parts[0]), 0)} " + " ".join(parts[1:]))
        if lines:
            dst.write_text("\n".join(lines), encoding="utf-8")
