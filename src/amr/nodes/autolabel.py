#!/usr/bin/env python3
"""Auto-label a folder of images with Grounding DINO and build a YOLO dataset.

Example:
  python3 auto_label_dino.py \
      --images ~/amr/captures \
      --out ~/datasets/amr_dataset \
      --classes "box=cardboard box" "chair" "table"

Each class is "name" or "name=prompt phrase". The prompt phrase is what
Grounding DINO is asked to find; the name is what YOLO will be trained on.
Output: images/{train,val}, labels/{train,val}, data.yaml, preview/.
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from torchvision.ops import nms
except Exception:  # torchvision missing: skip duplicate removal
    nms = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True, help='folder with .jpg/.png pictures')
    p.add_argument('--out', required=True, help='dataset output folder')
    p.add_argument('--classes', nargs='+', required=True,
                   help='"name" or "name=prompt phrase"')
    p.add_argument('--model', default='IDEA-Research/grounding-dino-base',
                   help='or IDEA-Research/grounding-dino-tiny (faster)')
    p.add_argument('--box-threshold', type=float, default=0.35)
    p.add_argument('--text-threshold', type=float, default=0.25)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--min-box-px', type=float, default=8)
    p.add_argument('--max-box-frac', type=float, default=0.9,
                   help='drop boxes covering more than this fraction of the image')
    p.add_argument('--val-fraction', type=float, default=0.15)
    p.add_argument('--keep-empty', type=float, default=0.1,
                   help='fraction of images with no detections to keep')
    p.add_argument('--preview', type=int, default=60,
                   help='number of preview images with boxes drawn')
    return p.parse_args()


def post_process(processor, outputs, input_ids, box_th, text_th, size):
    """Handle the argument name change between transformers versions."""
    try:
        return processor.post_process_grounded_object_detection(
            outputs, input_ids, threshold=box_th,
            text_threshold=text_th, target_sizes=[size])[0]
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs, input_ids, box_threshold=box_th,
            text_threshold=text_th, target_sizes=[size])[0]


def match_class(label, prompts):
    """Map the text DINO returned to one class index, or None if unclear."""
    label = str(label).lower().strip()
    hits = [i for i, ph in enumerate(prompts) if ph in label]
    if not hits:
        return None
    # Keep the longest matching phrase; reject if two different classes tie
    best = max(len(prompts[i]) for i in hits)
    top = [i for i in hits if len(prompts[i]) == best]
    return top[0] if len(top) == 1 else None


def main():
    args = parse_args()
    names, prompts = [], []
    for c in args.classes:
        name, _, phrase = c.partition('=')
        names.append(name.strip())
        prompts.append((phrase or name).strip().lower())

    out = os.path.expanduser(args.out)
    for split in ('train', 'val'):
        os.makedirs(f'{out}/images/{split}', exist_ok=True)
        os.makedirs(f'{out}/labels/{split}', exist_ok=True)
    os.makedirs(f'{out}/preview', exist_ok=True)

    files = []
    for ext in ('jpg', 'jpeg', 'png'):
        files += glob.glob(os.path.join(os.path.expanduser(args.images), f'*.{ext}'))
    files.sort()
    if not files:
        raise SystemExit('No images found in --images folder.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {args.model} on {device} ...')
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    model.eval()

    text = '. '.join(prompts) + '.'
    print(f'Prompt: {text!r} | {len(files)} images')

    stats = {'images': 0, 'with_boxes': 0, 'boxes': 0, 'kept': 0}
    per_class = [0] * len(names)
    previews = 0

    for n, path in enumerate(files, 1):
        image = Image.open(path).convert('RGB')
        w, h = image.size

        inputs = processor(images=image, text=text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = post_process(processor, outputs, inputs.input_ids,
                           args.box_threshold, args.text_threshold, (h, w))

        labels = res.get('text_labels', res['labels'])
        boxes, scores, cls = [], [], []
        for b, s, lab in zip(res['boxes'].cpu().tolist(),
                             res['scores'].cpu().tolist(), labels):
            ci = match_class(lab, prompts)
            if ci is None:
                continue
            x1, y1, x2, y2 = b
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < args.min_box_px or bh < args.min_box_px:
                continue
            if bw * bh > args.max_box_frac * w * h:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(s)
            cls.append(ci)

        # Remove duplicate boxes of the same class
        if boxes and nms is not None:
            keep = []
            t_boxes = torch.tensor(boxes)
            t_scores = torch.tensor(scores)
            t_cls = torch.tensor(cls)
            for c in set(cls):
                idx = (t_cls == c).nonzero().flatten()
                k = nms(t_boxes[idx], t_scores[idx], args.nms_iou)
                keep += idx[k].tolist()
            boxes = [boxes[i] for i in keep]
            cls = [cls[i] for i in keep]
            scores = [scores[i] for i in keep]

        stats['images'] += 1
        if boxes:
            stats['with_boxes'] += 1
            stats['boxes'] += len(boxes)
        elif random.random() > args.keep_empty:
            continue

        split = 'val' if random.random() < args.val_fraction else 'train'
        stem = os.path.splitext(os.path.basename(path))[0]
        shutil.copy(path, f'{out}/images/{split}/{stem}{os.path.splitext(path)[1]}')
        with open(f'{out}/labels/{split}/{stem}.txt', 'w') as f:
            for (x1, y1, x2, y2), c in zip(boxes, cls):
                per_class[c] += 1
                f.write(f'{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} '
                        f'{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n')
        stats['kept'] += 1

        if boxes and previews < args.preview:
            img = cv2.imread(path)
            for (x1, y1, x2, y2), c, s in zip(boxes, cls, scores):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(img, f'{names[c]} {s:.2f}', (int(x1), max(int(y1) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(f'{out}/preview/{stem}.jpg', img)
            previews += 1

        if n % 20 == 0 or n == len(files):
            print(f'{n}/{len(files)} done | images with boxes: '
                  f'{stats["with_boxes"]} | boxes: {stats["boxes"]}')

    with open(f'{out}/data.yaml', 'w') as f:
        f.write(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, n in enumerate(names):
            f.write(f'  {i}: {n}\n')

    print('\nDone.')
    print(f'Images saved: {stats["kept"]} (of {stats["images"]})')
    for n, c in zip(names, per_class):
        print(f'  {n}: {c} boxes')
    print(f'Check the labels in: {out}/preview')
    print(f'Train with: python3 train_gazebo_yolo.py --data {out}/data.yaml')


if __name__ == '__main__':
    main()