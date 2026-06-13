"""
Turn bus-camera footage into frames ready for labelling.

Two input modes:
  * a video file  -> samples one frame every N seconds
  * a folder of images (e.g. saved ESP32-CAM captures) -> copies/renames them

Frames land in a split's images/ dir with a consistent prefix so they never
collide with the public datasets. Label them afterwards (see README.md), then
either add bus_real as a dataset entry for fine-tuning or keep test/ as a
held-out bus-only metric.

Usage:
  python extract_frames.py --video ride1.mp4 --split train --every 2
  python extract_frames.py --images-dir ./esp32_dump --split test
"""
import argparse
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def from_video(video: Path, out_dir: Path, prefix: str, every_s: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every_s)))
    saved = idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out = out_dir / f"{prefix}_{video.stem}_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"{video.name}: saved {saved} frames (1 every {every_s}s, fps~{fps:.0f})")
    return saved


def from_images(src: Path, out_dir: Path, prefix: str) -> int:
    saved = 0
    for f in sorted(src.iterdir()):
        if f.suffix.lower() in EXT:
            shutil.copy2(f, out_dir / f"{prefix}_{f.name}")
            saved += 1
    print(f"{src}: copied {saved} images")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, help="Video file to sample frames from")
    ap.add_argument("--images-dir", type=Path, help="Folder of existing frames/captures")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="test")
    ap.add_argument("--every", type=float, default=2.0, help="Seconds between sampled frames (video mode)")
    ap.add_argument("--prefix", default="bus", help="Filename prefix to avoid collisions")
    args = ap.parse_args()

    out_dir = ROOT / args.split / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        from_video(args.video, out_dir, args.prefix, args.every)
    elif args.images_dir:
        from_images(args.images_dir, out_dir, args.prefix)
    else:
        raise SystemExit("Provide --video or --images-dir")

    print(f"Frames in: {out_dir}\nNext: label them, writing YOLO .txt into the sibling labels/ dir (class 0 = head).")


if __name__ == "__main__":
    main()
