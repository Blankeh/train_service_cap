"""Unit tests for PreprocessService."""

from pathlib import Path

import yaml
import pytest

from src.core.config import DatasetConfig, DatasetEntry, DatasetOutputConfig
from src.services.preprocess_service import PreprocessService


def _make_config(ds1_root: Path, ds2_root: Path, out: Path) -> DatasetConfig:
    return DatasetConfig(
        datasets=[
            DatasetEntry(
                path=str(ds1_root),
                name="ds1",
                class_map={0: 0},
                splits=["train", "valid", "test"],
            ),
            DatasetEntry(
                path=str(ds2_root),
                name="ds2",
                class_map={0: 0, 1: 0},
                splits=["train"],
                auto_split={"train": 0.6, "valid": 0.2, "test": 0.2},
            ),
        ],
        output=DatasetOutputConfig(path=str(out), classes=["person"], seed=42),
    )


class TestPreprocessService:
    def test_creates_data_yaml(self, ds1_root, ds2_root, tmp_path):
        out = tmp_path / "merged"
        cfg = _make_config(ds1_root, ds2_root, out)
        yaml_path = PreprocessService(cfg).run()
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["nc"] == 1
        assert data["names"] == ["person"]

    def test_all_splits_created(self, ds1_root, ds2_root, tmp_path):
        out = tmp_path / "merged"
        cfg = _make_config(ds1_root, ds2_root, out)
        PreprocessService(cfg).run()
        for split in ("train", "valid", "test"):
            assert (out / split / "images").is_dir()
            assert (out / split / "labels").is_dir()

    def test_class_remapping(self, ds1_root, ds2_root, tmp_path):
        out = tmp_path / "merged"
        cfg = _make_config(ds1_root, ds2_root, out)
        PreprocessService(cfg).run()
        for lbl in (out / "train" / "labels").iterdir():
            for line in lbl.read_text().splitlines():
                cls = int(line.split()[0])
                assert cls == 0, f"Expected class 0 (person), got {cls} in {lbl.name}"

    def test_duplicate_removal(self, tmp_path):
        """Exact duplicate image in ds2 that already appeared in ds1 must be skipped."""
        ds1 = tmp_path / "ds1_dup"
        ds2 = tmp_path / "ds2_dup"
        for root, split in [(ds1, "train"), (ds2, "train")]:
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            # Write identical bytes — same hash
            (img_dir / "img.jpg").write_bytes(b"SAME_CONTENT")
            (lbl_dir / "img.txt").write_text("0 0.5 0.5 0.4 0.6\n")

        out = tmp_path / "merged_dup"
        cfg = DatasetConfig(
            datasets=[
                DatasetEntry(path=str(ds1), name="ds1", class_map={0: 0}, splits=["train"]),
                DatasetEntry(
                    path=str(ds2), name="ds2", class_map={0: 0}, splits=["train"],
                    auto_split={"train": 1.0, "valid": 0.0, "test": 0.0},
                ),
            ],
            output=DatasetOutputConfig(path=str(out), classes=["person"], seed=0),
        )
        PreprocessService(cfg).run()
        imgs = list((out / "train" / "images").iterdir())
        assert len(imgs) == 1, "Duplicate should have been removed"

    def test_images_without_labels_are_skipped(self, tmp_path):
        ds = tmp_path / "ds_nolabel"
        img_dir = ds / "train" / "images"
        lbl_dir = ds / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        (img_dir / "no_label.jpg").write_bytes(b"IMG_NO_LABEL")
        # Intentionally no matching .txt file

        out = tmp_path / "merged_nolabel"
        cfg = DatasetConfig(
            datasets=[
                DatasetEntry(
                    path=str(ds), name="ds", class_map={0: 0}, splits=["train"],
                    auto_split={"train": 1.0, "valid": 0.0, "test": 0.0},
                )
            ],
            output=DatasetOutputConfig(path=str(out), classes=["person"], seed=0),
        )
        PreprocessService(cfg).run()
        imgs = list((out / "train" / "images").glob("*"))
        assert len(imgs) == 0
