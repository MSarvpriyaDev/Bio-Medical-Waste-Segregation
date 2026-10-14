#!/usr/bin/env python3
"""Deployment-faithful evaluation of a .tflite detector for the Smart Mobile
Medical-Waste Collection and Segregation System (SIH 2026 - Team Pravah77).

tflite-model-maker reports COCO mAP (AP, AP50, per-class AP). What it does NOT
give you is what a BMW Rules 2016 4-bin sorter actually needs:

* per-class precision / recall at the thresholds you will run in production
* a class-level AND 4-bin (Type A/B/C/D) confusion matrix
* cross-hazard misclassification accounting (e.g., a sharp filed as recyclable)
* false positives per image on EMPTY bin frames - the direct measure of how much
  the model has memorised liners / background instead of objects

All of that runs against the exported .tflite, i.e. exactly what ships to the Pi 5.

    python metrics.py --model artifacts/export/deploy/model.tflite --split test \
                      --negatives-dir data/negatives
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import config as cfg
from dataset_prep import IMG_EXTS, Sample, imread, load_split
from tflite_utils import Detection, TFLiteDetector, nms

log = logging.getLogger("metrics")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every row of a (n,4) and b (m,4), boxes as x0,y0,x1,y1."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def collect_predictions(detector: TFLiteDetector, image_paths: Sequence[Path], low_thr: float = 0.05) -> List[List[Detection]]:
    """Run once at a low threshold; operating points are applied afterwards."""
    preds = []
    for p in image_paths:
        img = imread(p)
        if img is None:
            preds.append([])
            continue
        dets, _ = detector.infer(img, low_thr)
        preds.append(nms(dets, cfg.NMS_IOU))
    return preds


def score_predictions(samples: Sequence[Sample], preds: Sequence[Sequence[Detection]],
                      class_names: Sequence[str], score_thr: float, iou_thr: float = 0.5) -> dict:
    idx = {c: i for i, c in enumerate(class_names)}
    tp, fp, fn = Counter(), Counter(), Counter()
    names = list(class_names)
    confusion = {g: Counter() for g in names + ["background"]}
    bin_confusion = {b: Counter() for b in list(cfg.BMW_BINS) + ["background"]}

    for s, dets in zip(samples, preds):
        dets = sorted((d for d in dets if d.score >= score_thr), key=lambda d: -d.score)
        gt_boxes = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in s.boxes], np.float32).reshape(-1, 4)
        gt_cls = np.array([idx[b.name] for b in s.boxes], np.int64)

        # (1) class-aware matching -> TP / FP / FN per class
        used = np.zeros(len(gt_boxes), bool)
        for d in dets:
            ious = iou_matrix(np.array([d.box], np.float32), gt_boxes)[0] if len(gt_boxes) else np.zeros(0)
            ious = np.where((gt_cls == d.class_id) & ~used, ious, 0.0) if len(gt_boxes) else ious
            j = int(ious.argmax()) if len(ious) else -1
            cname = names[d.class_id] if 0 <= d.class_id < len(names) else "?"
            if j >= 0 and ious[j] >= iou_thr:
                used[j] = True
                tp[cname] += 1
            else:
                fp[cname] += 1
        for j in np.where(~used)[0]:
            fn[names[gt_cls[j]]] += 1

        # (2) class-agnostic matching -> who gets confused with whom (class & BMW bin level)
        used = np.zeros(len(gt_boxes), bool)
        for d in dets:
            pname = names[d.class_id] if 0 <= d.class_id < len(names) else "?"
            pbin = cfg.CLASS_TO_BIN.get(pname, cfg.FALLBACK_BIN)
            if len(gt_boxes):
                ious = iou_matrix(np.array([d.box], np.float32), gt_boxes)[0]
                ious = np.where(~used, ious, 0.0)
                j = int(ious.argmax())
                if ious[j] >= iou_thr:
                    used[j] = True
                    gname = names[gt_cls[j]]
                    gbin = cfg.CLASS_TO_BIN.get(gname, cfg.FALLBACK_BIN)
                    confusion[gname][pname] += 1
                    bin_confusion[gbin][pbin] += 1
                    continue
            confusion["background"][pname] += 1
            bin_confusion["background"][pbin] += 1
        for j in np.where(~used)[0]:
            gname = names[gt_cls[j]]
            gbin = cfg.CLASS_TO_BIN.get(gname, cfg.FALLBACK_BIN)
            confusion[gname]["missed"] += 1
            bin_confusion[gbin]["missed"] += 1

    per_class = {}
    for c in names:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[c] = {
            "bmw_bin": cfg.CLASS_TO_BIN.get(c, cfg.FALLBACK_BIN),
            "precision": p, "recall": r, "f1": f1,
            "tp": tp[c], "fp": fp[c], "fn": fn[c],
        }
    return {
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "per_class": per_class,
        "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "confusion": {g: dict(v) for g, v in confusion.items()},
        "bin_confusion": {b: dict(v) for b, v in bin_confusion.items()},
    }


def false_positives_on_negatives(detector: TFLiteDetector, image_paths: Sequence[Path],
                                 class_names: Sequence[str], thresholds: Sequence[float]) -> dict:
    """Empty-bin frames: every detection is a false positive."""
    preds = collect_predictions(detector, image_paths)
    out = {}
    for thr in thresholds:
        per_class: Counter = Counter()
        imgs_with_fp = 0
        total = 0
        for dets in preds:
            kept = [d for d in dets if d.score >= thr]
            total += len(kept)
            imgs_with_fp += bool(kept)
            per_class.update(class_names[d.class_id] for d in kept if 0 <= d.class_id < len(class_names))
        n = max(len(preds), 1)
        out[str(thr)] = {"images": len(preds), "fp_per_image": total / n,
                         "fraction_images_with_fp": imgs_with_fp / n, "fp_by_class": dict(per_class)}
    return out


def evaluate_model_file(model_path: Path, split_dir: Path, class_names: Sequence[str],
                        thresholds: Sequence[float] = cfg.EVAL_SCORE_THRESHOLDS,
                        negatives_dir: Optional[Path] = None, num_threads: int = cfg.NUM_THREADS) -> dict:
    detector = TFLiteDetector(model_path, num_threads)
    samples, _, _ = load_split(split_dir, class_names)
    preds = collect_predictions(detector, [s.image_path for s in samples])
    report = {"model": str(model_path), "split_dir": str(split_dir), "images": len(samples),
              "operating_points": {str(t): score_predictions(samples, preds, class_names, t) for t in thresholds}}
    if negatives_dir and Path(negatives_dir).is_dir():
        neg = sorted(p for p in Path(negatives_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        if neg:
            report["negatives"] = false_positives_on_negatives(detector, neg, class_names, thresholds)
    return report


def format_report(report: dict, primary: float = cfg.EVAL_PRIMARY_THRESHOLD) -> str:
    lines = [f"Model : {report['model']}", f"Images: {report['images']}"]
    key = str(primary) if str(primary) in report["operating_points"] else next(iter(report["operating_points"]))
    op = report["operating_points"][key]
    lines += ["", f"Per-class results @ score>={op['score_threshold']}, IoU>={op['iou_threshold']}",
              f"{'class':<12}{'bin':>5}{'precision':>10}{'recall':>9}{'f1':>7}{'TP':>6}{'FP':>6}{'FN':>6}"]
    for c, v in op["per_class"].items():
        bcode = v.get("bmw_bin", cfg.CLASS_TO_BIN.get(c, "?"))
        lines.append(f"{c:<12}{bcode:>5}{v['precision']:>10.3f}{v['recall']:>9.3f}{v['f1']:>7.3f}{v['tp']:>6}{v['fp']:>6}{v['fn']:>6}")
    lines.append(f"{'macro':<12}{'':>5}{op['macro_precision']:>10.3f}{op['macro_recall']:>9.3f}")
    lines += ["", "Operating points (macro P / macro R):"]
    for t, o in report["operating_points"].items():
        lines.append(f"  score>={t}: P={o['macro_precision']:.3f}  R={o['macro_recall']:.3f}")
    lines += ["", "Confusion (rows = ground truth, columns = predicted):"]
    cols = sorted({k for row in op["confusion"].values() for k in row})
    lines.append(f"{'':<12}" + "".join(f"{c[:9]:>10}" for c in cols))
    for g, row in op["confusion"].items():
        if row:
            lines.append(f"{g:<12}" + "".join(f"{row.get(c, 0):>10}" for c in cols))
    if "bin_confusion" in op:
        lines += ["", "BMW 2016 4-Bin Confusion (A=Yellow, B=Red, C=White, D=Blue):"]
        bcols = sorted({k for row in op["bin_confusion"].values() for k in row})
        lines.append(f"{'':<12}" + "".join(f"{c[:9]:>10}" for c in bcols))
        for b, row in op["bin_confusion"].items():
            if row:
                lines.append(f"{b:<12}" + "".join(f"{row.get(c, 0):>10}" for c in bcols))
    if "negatives" in report:
        lines += ["", "Empty-bin frames (every detection is a false positive):"]
        for t, v in report["negatives"].items():
            lines.append(f"  score>={t}: {v['fp_per_image']:.3f} FP/image, "
                         f"{100 * v['fraction_images_with_fp']:.1f}% of {v['images']} images, {v['fp_by_class']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--negatives-dir", type=Path, default=cfg.NEGATIVES_DIR)
    ap.add_argument("--threads", type=int, default=cfg.NUM_THREADS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    report = evaluate_model_file(args.model, cfg.PREPARED_SPLITS[args.split], cfg.CLASS_NAMES,
                                 negatives_dir=args.negatives_dir, num_threads=args.threads)
    print(format_report(report))
    out = args.out or cfg.REPORT_DIR / f"pr_{args.model.stem}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
