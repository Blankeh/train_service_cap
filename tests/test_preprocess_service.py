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
                box_style="head",
                splits=["train", "valid", "test"],
            ),
            DatasetEntry(
                path=str(ds2_root),
                name="ds2",
                class_map={0: 0, 1: 0},
                box_style="head",
                splits=["train"],
                auto_split={"train": 0.6, "valid": 0.2, "test": 0.2},
            ),
        ],
        output=DatasetOutputConfig(
            path=str(out), classes=["person"],
            target_box_style="head", seed=42,
        ),
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
                DatasetEntry(
                    path=str(ds1), name="ds1", class_map={0: 0},
                    box_style="head", splits=["train"],
                ),
                DatasetEntry(
                    path=str(ds2), name="ds2", class_map={0: 0},
                    box_style="head", splits=["train"],
                    auto_split={"train": 1.0, "valid": 0.0, "test": 0.0},
                ),
            ],
            output=DatasetOutputConfig(
                path=str(out), classes=["person"],
                target_box_style="head", seed=0,
            ),
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
                    path=str(ds), name="ds", class_map={0: 0},
                    box_style="head", splits=["train"],
                    auto_split={"train": 1.0, "valid": 0.0, "test": 0.0},
                )
            ],
            output=DatasetOutputConfig(
                path=str(out), classes=["person"],
                target_box_style="head", seed=0,
            ),
        )
        PreprocessService(cfg).run()
        imgs = list((out / "train" / "images").glob("*"))
        assert len(imgs) == 0


class TestPolygonConversion:
    """Tests for polygon → bbox conversion."""

    def test_polygon_converted_to_bbox(self, tmp_path):
        """A polygon label should produce a standard 5-value YOLO bbox line."""
        src = tmp_path / "poly.txt"
        dst = tmp_path / "out.txt"
        # Rectangle polygon: x=[0.3, 0.7], y=[0.2, 0.6]
        src.write_text("0 0.3 0.2 0.7 0.2 0.7 0.6 0.3 0.6\n")

        stats = PreprocessService._remap_label(
            src, dst, {0: 0}, source_style="body", target_style="body",
        )

        line = dst.read_text().strip()
        parts = line.split()
        assert len(parts) == 5, f"Expected 5 values (bbox), got {len(parts)}: {line}"
        assert stats["polygons_converted"] == 1

        # Check bbox values: cx=0.5, cy=0.4, w=0.4, h=0.4
        cls, cx, cy, w, h = int(parts[0]), *[float(p) for p in parts[1:]]
        assert cls == 0
        assert abs(cx - 0.5) < 0.001
        assert abs(cy - 0.4) < 0.001
        assert abs(w - 0.4) < 0.001
        assert abs(h - 0.4) < 0.001

    def test_bbox_passthrough(self, tmp_path):
        """A standard 5-value bbox should pass through unchanged when styles match."""
        src = tmp_path / "bbox.txt"
        dst = tmp_path / "out.txt"
        src.write_text("0 0.5 0.5 0.04 0.08\n")

        PreprocessService._remap_label(
            src, dst, {0: 0}, source_style="head", target_style="head",
        )

        line = dst.read_text().strip()
        parts = line.split()
        assert len(parts) == 5
        assert abs(float(parts[1]) - 0.5) < 0.001
        assert abs(float(parts[3]) - 0.04) < 0.001


class TestBoxStyleConversion:
    """Tests for body ↔ head normalization."""

    def test_body_to_head_shrinks_box(self, tmp_path):
        """Body→head should produce a smaller box anchored at the top."""
        src = tmp_path / "body.txt"
        dst = tmp_path / "out.txt"
        # Body bbox: cx=0.5, cy=0.5, w=0.2, h=0.5
        src.write_text("0 0.5 0.5 0.2 0.5\n")

        PreprocessService._remap_label(
            src, dst, {0: 0}, source_style="body", target_style="head",
        )

        line = dst.read_text().strip()
        parts = line.split()
        cx, cy, w, h = [float(p) for p in parts[1:]]
        # Head should be smaller
        assert w < 0.2, f"Head width {w} should be < body width 0.2"
        assert h < 0.5, f"Head height {h} should be < body height 0.5"
        # Head should be near the top of the body box
        body_top = 0.5 - 0.5 / 2  # = 0.25
        assert cy < 0.5, f"Head cy {cy} should be above body cy 0.5"
        assert abs(cy - (body_top + h / 2)) < 0.001

    def test_head_to_body_expands_box(self, tmp_path):
        """Head→body should produce a larger box expanding downward."""
        src = tmp_path / "head.txt"
        dst = tmp_path / "out.txt"
        # Head bbox: cx=0.5, cy=0.3, w=0.04, h=0.06
        src.write_text("0 0.5 0.3 0.04 0.06\n")

        PreprocessService._remap_label(
            src, dst, {0: 0}, source_style="head", target_style="body",
        )

        line = dst.read_text().strip()
        parts = line.split()
        cx, cy, w, h = [float(p) for p in parts[1:]]
        assert w > 0.04, f"Body width {w} should be > head width 0.04"
        assert h > 0.06, f"Body height {h} should be > head height 0.06"
        assert cy > 0.3, f"Body cy {cy} should be below head cy 0.3"

    def test_same_style_is_noop(self, tmp_path):
        """Same source and target style should not change the box."""
        src = tmp_path / "same.txt"
        dst = tmp_path / "out.txt"
        src.write_text("0 0.5 0.5 0.04 0.08\n")

        stats = PreprocessService._remap_label(
            src, dst, {0: 0}, source_style="head", target_style="head",
        )

        line = dst.read_text().strip()
        parts = line.split()
        assert abs(float(parts[3]) - 0.04) < 0.0001
        assert abs(float(parts[4]) - 0.08) < 0.0001
        assert stats["boxes_restyled"] == 0

    def test_polygon_body_to_head_full_pipeline(self, ds1_root, ds_body_root, tmp_path):
        """End-to-end: polygon body dataset + head dataset → all output is head-sized bboxes."""
        out = tmp_path / "merged_mixed"
        cfg = DatasetConfig(
            datasets=[
                DatasetEntry(
                    path=str(ds1_root), name="heads",
                    class_map={0: 0}, box_style="head",
                    splits=["train", "valid", "test"],
                ),
                DatasetEntry(
                    path=str(ds_body_root), name="bodies",
                    class_map={0: 0, 1: 0}, box_style="body",
                    splits=["train"],
                    auto_split={"train": 1.0, "valid": 0.0, "test": 0.0},
                ),
            ],
            output=DatasetOutputConfig(
                path=str(out), classes=["person"],
                target_box_style="head", seed=42,
            ),
        )
        PreprocessService(cfg).run()

        # Every label file should have exactly 5 values per line (no raw polygons)
        for split in ("train", "valid", "test"):
            lbl_dir = out / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl in lbl_dir.iterdir():
                for line_num, line in enumerate(lbl.read_text().splitlines(), 1):
                    parts = line.split()
                    assert len(parts) == 5, (
                        f"{lbl.name}:{line_num}: expected 5 values, got {len(parts)}"
                    )
                    # Class should always be 0 (person)
                    assert int(parts[0]) == 0

        # Body-sourced labels specifically should have been shrunk
        # The polygon was x=[0.4,0.6] y=[0.2,0.8] → body bbox w=0.2 h=0.6
        # After body→head: w = 0.2*0.5 = 0.10,  h = 0.6*0.2 = 0.12
        for lbl in (out / "train" / "labels").iterdir():
            if lbl.stem.startswith("body_"):
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    w, h = float(parts[3]), float(parts[4])
                    assert w < 0.15, f"{lbl.name}: width {w} too large for converted head box"
                    assert h < 0.15, f"{lbl.name}: height {h} too large for converted head box"
