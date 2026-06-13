"""
Convert CrowdHuman (ODGT + zipped images) into a YOLO head-detection dataset.

CrowdHuman ships:
  * CrowdHuman_train01/02/03.zip  -> 15,000 train images under  Images/<ID>.jpg
  * CrowdHuman_val.zip            ->  4,370 val   images under  Images/<ID>.jpg
  * annotation_train.odgt / annotation_val.odgt  (one JSON record per line)

Each gtbox carries hbox=[x,y,w,h] (head, absolute px, top-left origin). We keep
the head box for tag=="person" with head_attr.ignore!=1, normalise to YOLO
cx,cy,w,h (class 0), and write one .txt per image. Filenames are sanitised
(the IDs contain a comma) so downstream tools don't choke.

Output:
  <OUT>/train/images/*.jpg + <OUT>/train/labels/*.txt
  <OUT>/valid/images/*.jpg + <OUT>/valid/labels/*.txt
"""
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

DL = Path(r"C:/Users/blank/Downloads")
OUT = Path(r"C:/Users/blank/Downloads/crowdhuman_yolo")

SPLITS = {
    "train": {
        "zips": ["CrowdHuman_train01.zip", "CrowdHuman_train02.zip", "CrowdHuman_train03.zip"],
        "odgt": "annotation_train.odgt",
    },
    "valid": {
        "zips": ["CrowdHuman_val.zip"],
        "odgt": "annotation_val.odgt",
    },
}


def load_annotations(odgt_path: Path) -> dict[str, list]:
    ann = {}
    with open(odgt_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ann[rec["ID"]] = rec.get("gtboxes", [])
    return ann


def to_yolo_heads(gtboxes: list, w: int, h: int) -> list[str]:
    """Return YOLO label lines (class 0) from head boxes, clamped to the image."""
    lines = []
    for gt in gtboxes:
        if gt.get("tag") != "person":
            continue  # skip 'mask' ignore regions
        if gt.get("head_attr", {}).get("ignore", 0) == 1:
            continue  # unreliable head annotation
        hbox = gt.get("hbox")
        if not hbox or len(hbox) != 4:
            continue
        x, y, bw, bh = hbox
        # clamp box to image bounds
        x1 = max(0, min(x, w))
        y1 = max(0, min(y, h))
        x2 = max(0, min(x + bw, w))
        y2 = max(0, min(y + bh, h))
        bw_c, bh_c = x2 - x1, y2 - y1
        if bw_c <= 1 or bh_c <= 1:
            continue  # degenerate after clamping
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        nw = bw_c / w
        nh = bh_c / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_split(split: str, spec: dict) -> None:
    ann = load_annotations(DL / spec["odgt"])
    img_dir = OUT / split / "images"
    lbl_dir = OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_img = n_lbl = n_box = n_skip = 0
    for zname in spec["zips"]:
        zpath = DL / zname
        if not zpath.exists():
            print(f"  WARNING missing zip: {zname}", flush=True)
            continue
        with zipfile.ZipFile(zpath) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
            print(f"  {zname}: {len(members)} images", flush=True)
            for i, m in enumerate(members):
                rec_id = Path(m).stem                 # "273271,abc..."
                safe = rec_id.replace(",", "_")        # filesystem/parse-safe
                data = zf.read(m)
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
                except Exception:
                    n_skip += 1
                    continue
                if rec_id not in ann:
                    n_skip += 1
                    continue
                lines = to_yolo_heads(ann[rec_id], w, h)
                if not lines:
                    n_skip += 1
                    continue
                (img_dir / f"{safe}.jpg").write_bytes(data)
                (lbl_dir / f"{safe}.txt").write_text("\n".join(lines), encoding="utf-8")
                n_img += 1
                n_lbl += 1
                n_box += len(lines)
                if (i + 1) % 2000 == 0:
                    print(f"    {zname}: {i + 1} processed", flush=True)
    print(f"[{split}] images={n_img} labels={n_lbl} boxes={n_box} skipped={n_skip}", flush=True)


def main() -> None:
    for split, spec in SPLITS.items():
        print(f"== Converting {split} ==", flush=True)
        convert_split(split, spec)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
