# bus_real — your own bus-camera head data

In-domain frames from the **actual ESP32-CAM mounting angle**. A few hundred
labelled frames here are worth more than any public dataset for your specific
deployment, because nothing else matches your camera height, lens, lighting and
crowd density.

Use it for **(a)** a held-out bus-only test set (honest deployment metric) and
**(b)** second-stage fine-tuning after the big passenger+CrowdHuman pretrain.

## Layout

```
bus_real/
  train/{images,labels}    # optional — for fine-tuning
  valid/{images,labels}    # optional
  test/{images,labels}     # priority — your real deployment metric
```

Labels are YOLO format, **class 0 = head** (single class, matches the merged
dataset's `person`/head box style).

## 1. Get frames

From a recorded bus ride (samples 1 frame every 2 s):
```
python extract_frames.py --video ride1.mp4 --split test --every 2
```
From a folder of saved ESP32-CAM captures:
```
python extract_frames.py --images-dir ./esp32_dump --split test
```

Aim for variety: full/empty bus, standing crowds, day/night, glare. ~150–300
frames is a solid starting test set; 500+ if you also want to fine-tune.

## 2. Label the heads

Draw a tight box around each **visible head** (one class). Tools:
- **Label Studio** (`pip install label-studio`) — export "YOLO"
- **LabelImg** (`pip install labelImg`) — set format to YOLO, single class
- **Roboflow** — annotate in-browser, export YOLOv8

Put each `*.txt` next to its image in the sibling `labels/` dir. Lines look like:
```
0 0.512 0.337 0.041 0.066
```

## 3. Use it

**As a held-out bus metric** (recommended first):
```
yolo val model=runs/.../best.pt data=bus_real/bus_eval.yaml
```
where `bus_eval.yaml` points `val:` at `bus_real/test/images`. This tells you
real bus performance, separate from the mixed passenger+CrowdHuman val set.

**As fine-tuning data:** add an entry to `configs/dataset.yaml`:
```yaml
  - path: "C:/Users/blank/projects/capstone/dev/train_service_cap/bus_real"
    name: bus_real
    class_map: {0: 0}
    box_style: head
    splits: [train, valid, test]
```
then re-run the pipeline (without `--skip-preprocess`). For best results,
fine-tune the pretrained model on bus_real specifically rather than just
diluting it into the 46k mixed set.

## Note
Images are git-ignored (large/private). Labels are committed so the annotation
work is tracked.
