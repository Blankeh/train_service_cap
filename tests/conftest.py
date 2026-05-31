import pytest
from pathlib import Path


def make_image(path: Path, seed: int = 0) -> Path:
    """Write a minimal synthetic image file (unique bytes per seed)."""
    path.write_bytes(b"FAKE_IMG_" + str(seed).encode())
    return path


def make_label(path: Path, class_id: int = 0) -> Path:
    """Write a YOLO label file with one detection (bbox format)."""
    path.write_text(f"{class_id} 0.5 0.5 0.4 0.6\n", encoding="utf-8")
    return path


def make_polygon_label(path: Path, class_id: int = 0) -> Path:
    """Write a YOLO label file with one detection (polygon/segmentation format)."""
    # A simple polygon that encloses a body-sized region
    path.write_text(
        f"{class_id} 0.40 0.20 0.60 0.20 0.60 0.80 0.40 0.80\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def ds1_root(tmp_path: Path) -> Path:
    """Synthetic dataset with train/valid/test splits, class: head (0)."""
    for split in ("train", "valid", "test"):
        img_dir = tmp_path / "ds1" / split / "images"
        lbl_dir = tmp_path / "ds1" / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(3):
            stem = f"{split}_img_{i}"
            make_image(img_dir / f"{stem}.jpg", seed=hash(f"ds1_{split}_{i}") % 10000)
            make_label(lbl_dir / f"{stem}.txt", class_id=0)
    return tmp_path / "ds1"


@pytest.fixture
def ds2_root(tmp_path: Path) -> Path:
    """Synthetic dataset with train only, classes: man (0), woman (1)."""
    img_dir = tmp_path / "ds2" / "train" / "images"
    lbl_dir = tmp_path / "ds2" / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(5):
        stem = f"train_img_{i}"
        make_image(img_dir / f"{stem}.jpg", seed=hash(f"ds2_train_{i}") % 10000)
        make_label(lbl_dir / f"{stem}.txt", class_id=i % 2)  # alternates man/woman
    return tmp_path / "ds2"


@pytest.fixture
def ds_body_root(tmp_path: Path) -> Path:
    """Synthetic dataset with train only, body polygon annotations."""
    img_dir = tmp_path / "ds_body" / "train" / "images"
    lbl_dir = tmp_path / "ds_body" / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(4):
        stem = f"body_img_{i}"
        make_image(img_dir / f"{stem}.jpg", seed=hash(f"ds_body_{i}") % 10000)
        make_polygon_label(lbl_dir / f"{stem}.txt", class_id=i % 2)
    return tmp_path / "ds_body"

