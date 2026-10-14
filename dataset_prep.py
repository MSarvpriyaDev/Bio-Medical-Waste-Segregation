#!/usr/bin/env python3
"""Roboflow Pascal-VOC verification, sanitising and offline augmentation for the
Smart Mobile Medical-Waste Collection and Segregation System (SIH 2026 - Team Pravah77).

Why Pascal VOC?  tflite-model-maker's detector loader reads Pascal VOC XML
(``DataLoader.from_pascal_voc``) or CSV.  It has no COCO-JSON loader, and
TFRecord only helps if you leave Model Maker for the TF Object Detection API.
So: export from Roboflow as **Pascal VOC**, with "Stretch to 640x640" (square)
as the only preprocessing step and **no Roboflow augmentation** (we do it here,
class-aware across all 4 BMW 2016 waste bin categories, on the train split only).

Addresses SIH 2026 PDF Technical Challenges:
- Confusion in 2D image processing due to large dumps / mixed waste: synthesises
  multi-object 4-type mixed-waste training scenes so the CNN and separation-line
  coordinator can isolate composite waste items.
- Optical distortion on transparent and reflective surfaces (glassware, vials,
  syringes, wet bio-fluids): applies specular glare and colour-cast augmentations.

Sub-commands
------------
    python dataset_prep.py discover    # list class names found in the XML files
    python dataset_prep.py verify      # integrity report, BMW bin balance, split leakage
    python dataset_prep.py build       # verify + write data/prepared/{train,val,test}
        --no-augment                   # sanitise only
        --bg-dir data/backgrounds      # empty-bin photos for copy-paste synthesis

Only OpenCV + NumPy + Pillow are used (no albumentations) so this runs in the
same old-NumPy environment that tflite-model-maker forces on you.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import config as cfg

log = logging.getLogger("dataset_prep")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Data structures + VOC parsing
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: int = 0


@dataclass
class Sample:
    image_path: Path
    xml_path: Path
    width: int
    height: int
    boxes: List[Box]


def imread(path: Path) -> Optional[np.ndarray]:
    """BGR read that ignores EXIF rotation, matching TensorFlow's decoder."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def parse_voc_xml(xml_path: Path) -> Tuple[Optional[str], List[Box]]:
    root = ET.parse(str(xml_path)).getroot()
    boxes: List[Box] = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            coords = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        try:
            difficult = int(float(obj.findtext("difficult") or 0))
        except ValueError:
            difficult = 0
        boxes.append(Box((obj.findtext("name") or "").strip(), *coords, difficult=difficult))
    return root.findtext("filename"), boxes


def locate_image(xml_path: Path, filename: Optional[str]) -> Optional[Path]:
    cands = []
    for ext in IMG_EXTS:
        cands += [xml_path.with_suffix(ext), xml_path.with_suffix(ext.upper())]
    if filename:
        cands.append(xml_path.parent / filename)
    return next((c for c in cands if c.exists()), None)


def write_voc_xml(path: Path, image_name: str, width: int, height: int, boxes: Sequence[Box]) -> None:
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = path.parent.name
    ET.SubElement(ann, "filename").text = image_name
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"
    for b in boxes:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = b.name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = str(int(b.difficult))
        bb = ET.SubElement(obj, "bndbox")
        for key, val in (("xmin", b.xmin), ("ymin", b.ymin), ("xmax", b.xmax), ("ymax", b.ymax)):
            ET.SubElement(bb, key).text = str(int(round(val)))
    ET.ElementTree(ann).write(str(path), encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------------------- #
# Loading + verification
# --------------------------------------------------------------------------- #
def load_split(split_dir: Path, class_names: Sequence[str]) -> Tuple[List[Sample], Dict[str, List[str]], Counter]:
    """Parse one split. Returns (clean samples, issues by kind, unknown-class counter).

    Boxes are clipped to the image, degenerate boxes dropped, and the true image
    size (not the size claimed by the XML) is used - Model Maker normalises box
    coordinates with the XML <size>, so a wrong value silently corrupts labels.
    """
    known = set(class_names)
    issues: Dict[str, List[str]] = defaultdict(list)
    unknown: Counter = Counter()
    samples: List[Sample] = []
    xml_paths = sorted(split_dir.glob("*.xml"))

    for xml_path in xml_paths:
        try:
            fname, raw_boxes = parse_voc_xml(xml_path)
        except ET.ParseError:
            issues["unparseable_xml"].append(xml_path.name)
            continue
        img_path = locate_image(xml_path, fname)
        if img_path is None:
            issues["missing_image"].append(xml_path.name)
            continue
        try:
            with Image.open(img_path) as im:
                width, height = im.size  # PIL does not apply EXIF rotation
                if im.getexif().get(0x0112, 1) != 1:
                    issues["exif_orientation_tag"].append(img_path.name)
        except Exception:  # noqa: BLE001 - any decoder failure means "unusable"
            issues["unreadable_image"].append(img_path.name)
            continue
        if width != height:
            issues["non_square_image"].append(img_path.name)

        clean: List[Box] = []
        for b in raw_boxes:
            if b.name not in known:
                unknown[b.name] += 1
                continue
            x0, x1 = sorted((b.xmin, b.xmax))
            y0, y1 = sorted((b.ymin, b.ymax))
            x0, x1 = max(0.0, x0), min(float(width), x1)
            y0, y1 = max(0.0, y0), min(float(height), y1)
            if (x1 - x0) < cfg.MIN_BOX_PX or (y1 - y0) < cfg.MIN_BOX_PX:
                issues["degenerate_box"].append(f"{xml_path.name}:{b.name}")
                continue
            clean.append(Box(b.name, x0, y0, x1, y1, b.difficult))
        if not clean:
            issues["no_objects"].append(img_path.name)
        samples.append(Sample(img_path, xml_path, width, height, clean))

    xml_stems = {p.stem for p in xml_paths}
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS and p.stem not in xml_stems:
            issues["image_without_xml"].append(p.name)
    return samples, dict(issues), unknown


def class_statistics(samples: Sequence[Sample], class_names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    inst, imgs = Counter(), Counter()
    for s in samples:
        inst.update(b.name for b in s.boxes)
        imgs.update({b.name for b in s.boxes})
    return {c: {"instances": inst[c], "images": imgs[c]} for c in class_names}


def bmw_bin_statistics(samples: Sequence[Sample]) -> Dict[str, int]:
    """Count total object instances mapped to each BMW Rules 2016 bin (A/B/C/D)."""
    counts = {b: 0 for b in cfg.BMW_BINS}
    for s in samples:
        for bx in s.boxes:
            bcode = cfg.CLASS_TO_BIN.get(bx.name, cfg.FALLBACK_BIN)
            counts[bcode] = counts.get(bcode, 0) + 1
    return counts


def imbalance_ratio(stats: Dict[str, Dict[str, int]]) -> float:
    counts = [v["instances"] for v in stats.values() if v["instances"] > 0]
    return (max(counts) / min(counts)) if counts else float("inf")


def source_id(image_name: str) -> str:
    """Roboflow names files '<source>_jpg.rf.<hash>.jpg'; the part before '.rf.' is the source."""
    return re.split(r"\.rf\.", Path(image_name).stem)[0]


def dhash(path: Path, size: int = 8) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return 0
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = np.packbits((small[:, 1:] > small[:, :-1]).flatten())
    return int(np.frombuffer(bits.tobytes(), dtype=">u8")[0])


def find_split_leakage(split_samples: Dict[str, List[Sample]], max_hamming: int) -> List[Tuple[str, str, str, str, str]]:
    """Return (split_a, file_a, split_b, file_b, reason) for near-duplicates across splits."""
    leaks: List[Tuple[str, str, str, str, str]] = []
    names = [n for n in ("train", "val", "test") if split_samples.get(n)]
    hashes = {n: np.array([dhash(s.image_path) for s in split_samples[n]], dtype=np.uint64) for n in names}
    sources = {n: defaultdict(list) for n in names}
    for n in names:
        for s in split_samples[n]:
            sources[n][source_id(s.image_path.name)].append(s.image_path.name)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for sid in set(sources[a]) & set(sources[b]):
                leaks.append((a, sources[a][sid][0], b, sources[b][sid][0], "same Roboflow source"))
            ha, hb = hashes[a], hashes[b]
            for start in range(0, len(hb), 256):
                chunk = hb[start:start + 256]
                x = np.bitwise_xor(chunk[:, None], ha[None, :])
                dist = _POP8[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)
                for ci, aj in np.argwhere(dist <= max_hamming):
                    leaks.append((a, split_samples[a][aj].image_path.name,
                                  b, split_samples[b][start + ci].image_path.name,
                                  f"near-duplicate (dHash distance {int(dist[ci, aj])})"))
    return leaks


def discover_classes(split_dirs: Dict[str, Path]) -> Counter:
    found: Counter = Counter()
    for d in split_dirs.values():
        for xml_path in sorted(d.glob("*.xml")) if d.exists() else []:
            try:
                found.update(b.name for b in parse_voc_xml(xml_path)[1])
            except ET.ParseError:
                pass
    return found


def verify_dataset(splits: Dict[str, Path], class_names: Sequence[str]) -> Tuple[Dict[str, List[Sample]], dict, List[str]]:
    """Full verification. Returns (samples per split, JSON-able report, fatal errors)."""
    errors: List[str] = []
    report: dict = {"splits": {}, "leakage": []}
    all_samples: Dict[str, List[Sample]] = {}

    for name, d in splits.items():
        if not d.is_dir():
            errors.append(f"split '{name}' directory not found: {d}")
            continue
        samples, issues, unknown = load_split(d, class_names)
        all_samples[name] = samples
        stats = class_statistics(samples, class_names)
        bin_stats = bmw_bin_statistics(samples)
        entry = {
            "images": len(samples),
            "objects": sum(len(s.boxes) for s in samples),
            "class_stats": stats,
            "bmw_bin_stats": bin_stats,
            "imbalance_ratio": imbalance_ratio(stats),
            "issues": {k: len(v) for k, v in issues.items()},
            "issue_examples": {k: v[:5] for k, v in issues.items()},
            "unknown_classes": dict(unknown),
        }
        report["splits"][name] = entry
        if unknown:
            errors.append(f"[{name}] class names in XML but not in config.CLASS_NAMES: {dict(unknown)}")
        if not samples:
            errors.append(f"[{name}] no annotated images found in {d}")
        if name != "train":
            for c, v in stats.items():
                if v["instances"] == 0:
                    log.warning("[%s] class '%s' has 0 instances - its mAP will be undefined", name, c)
        log.info("[%s] %d images, %d objects, imbalance %.1f:1 | BMW Bins A/B/C/D: %s",
                 name, entry["images"], entry["objects"], entry["imbalance_ratio"], bin_stats)
        for c, v in stats.items():
            bcode = cfg.CLASS_TO_BIN.get(c, cfg.FALLBACK_BIN)
            log.info("    %-12s (Bin %s) %5d objects in %5d images", c, bcode, v["instances"], v["images"])
        for kind, items in issues.items():
            level = logging.WARNING if kind not in ("unparseable_xml", "missing_image", "unreadable_image") else logging.ERROR
            log.log(level, "[%s] %s: %d (e.g. %s)", name, kind, len(items), ", ".join(items[:3]))
        if issues.get("non_square_image"):
            log.warning("[%s] non-square images: the TFLite model stretches frames to %dx%d at inference, "
                        "but Model Maker letterboxes non-square training images. Re-export from Roboflow "
                        "with Resize = 'Stretch to 640x640'.", name, cfg.INPUT_SIZE, cfg.INPUT_SIZE)

    if len(all_samples) == len(splits):
        leaks = find_split_leakage(all_samples, cfg.NEAR_DUP_HAMMING)
        report["leakage"] = [list(x) for x in leaks[:200]]
        if leaks:
            log.warning("LEAKAGE: %d near-duplicate pairs across splits (e.g. %s). Metrics will be inflated; "
                        "re-split by capture session, not by random frame.", len(leaks), leaks[0])
        else:
            log.info("No cross-split leakage detected (same-source + dHash<=%d).", cfg.NEAR_DUP_HAMMING)
    return all_samples, report, errors


# --------------------------------------------------------------------------- #
# Class imbalance: repeat-factor sampling
# --------------------------------------------------------------------------- #
def repeat_factors(samples: Sequence[Sample], class_names: Sequence[str], threshold: float) -> Tuple[List[float], Dict[str, float]]:
    n = max(len(samples), 1)
    img_count: Counter = Counter()
    for s in samples:
        img_count.update({b.name for b in s.boxes})
    r_c = {c: max(1.0, math.sqrt(threshold / (img_count[c] / n))) for c in class_names if img_count[c] > 0}
    per_image = [max((r_c[b.name] for b in s.boxes), default=1.0) for s in samples]
    return per_image, r_c


def plan_copies(samples: Sequence[Sample], class_names: Sequence[str], rng: np.random.Generator) -> List[int]:
    factors, r_c = repeat_factors(samples, class_names, cfg.RFS_THRESHOLD)
    log.info("repeat factors: %s", {c: round(v, 2) for c, v in r_c.items()})
    copies = []
    for r in factors:
        x = cfg.AUG_BASE_COPIES * r
        n = int(x) + (1 if rng.random() < x - int(x) else 0)
        copies.append(min(n, cfg.AUG_MAX_COPIES))
    return copies


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def _overlap(cand: np.ndarray, boxes: np.ndarray) -> float:
    """Largest of inter/area(cand) and inter/area(box) over all boxes."""
    if len(boxes) == 0:
        return 0.0
    ix0 = np.maximum(cand[0], boxes[:, 0]); iy0 = np.maximum(cand[1], boxes[:, 1])
    ix1 = np.minimum(cand[2], boxes[:, 2]); iy1 = np.minimum(cand[3], boxes[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    a_c = max((cand[2] - cand[0]) * (cand[3] - cand[1]), 1e-6)
    a_b = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    return float(max((inter / a_c).max(), (inter / a_b).max()))


class CropBank:
    """Object crops (from TRAIN images only) used for copy-paste; rare classes are drawn more often."""

    def __init__(self, samples: Sequence[Sample], class_names: Sequence[str], pad: float = 0.06, max_crops: int = 3000):
        _, r_c = repeat_factors(samples, class_names, cfg.RFS_THRESHOLD)
        self.crops: List[np.ndarray] = []
        self.names: List[str] = []
        weights: List[float] = []
        for s in samples:
            if len(self.crops) >= max_crops:
                break
            img = imread(s.image_path)
            if img is None:
                continue
            for b in s.boxes:
                bw, bh = b.xmax - b.xmin, b.ymax - b.ymin
                x0, x1 = int(max(0, b.xmin - pad * bw)), int(min(s.width, b.xmax + pad * bw))
                y0, y1 = int(max(0, b.ymin - pad * bh)), int(min(s.height, b.ymax + pad * bh))
                if x1 - x0 < 12 or y1 - y0 < 12:
                    continue
                self.crops.append(img[y0:y1, x0:x1].copy())
                self.names.append(b.name)
                weights.append(r_c.get(b.name, 1.0))
        w = np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum() if len(w) else w

    def __len__(self) -> int:
        return len(self.crops)

    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, str]:
        i = int(rng.choice(len(self.crops), p=self.p))
        return self.crops[i], self.names[i]


def paste_crops(canvas: np.ndarray, boxes: np.ndarray, labels: List[str], bank: CropBank,
                rng: np.random.Generator, n_paste: int, max_overlap: float = 0.15
                ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Feather-blend crops onto the canvas. New boxes are the exact pasted rectangles."""
    h, w = canvas.shape[:2]
    canvas = canvas.copy()
    boxes = boxes.reshape(-1, 4).astype(np.float32)
    for _ in range(n_paste):
        if len(bank) == 0:
            break
        crop, name = bank.sample(rng)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        crop = np.ascontiguousarray(np.rot90(crop, int(rng.integers(0, 4))))
        ch, cw = crop.shape[:2]
        s = min(rng.uniform(0.7, 1.3), 0.6 * h / ch, 0.6 * w / cw)
        nh, nw = max(8, int(ch * s)), max(8, int(cw * s))
        if nh >= h or nw >= w:
            continue
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        for _try in range(10):
            x0 = int(rng.integers(0, w - nw + 1))
            y0 = int(rng.integers(0, h - nh + 1))
            cand = np.array([x0, y0, x0 + nw, y0 + nh], np.float32)
            if _overlap(cand, boxes) > max_overlap:
                continue
            mask = np.zeros((nh, nw), np.float32)
            mask[2:-2, 2:-2] = 1.0
            k = max(3, (min(nh, nw) // 6) | 1)
            mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
            roi = canvas[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
            canvas[y0:y0 + nh, x0:x0 + nw] = (mask * crop + (1.0 - mask) * roi).astype(np.uint8)
            boxes = np.vstack([boxes, cand[None]])
            labels = labels + [name]
            break
    return canvas, boxes, labels


class WasteBinAugmenter:
    """Bounding-box-aware augmentation tuned for a fixed top-down camera over a
    medical-grade hydrophobic non-stick tilting dished plate.
    """

    def __init__(self, rng: np.random.Generator, params: Optional[dict] = None):
        self.rng = rng
        self.p = dict(cfg.AUGMENTATION)
        if params:
            self.p.update(params)
        self._grid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    # ----- helpers
    def _chance(self, key: str) -> bool:
        return bool(self.rng.random() < self.p[key])

    def _grid(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        if (h, w) not in self._grid_cache:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            self._grid_cache = {(h, w): (yy, xx)}   # keep a single entry
        return self._grid_cache[(h, w)]

    # ----- geometric (boxes move with the pixels)
    def _flip_rot90(self, img: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        if self._chance("p_hflip"):
            img = img[:, ::-1]
            if len(boxes):
                boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
        if self._chance("p_vflip"):
            img = img[::-1]
            if len(boxes):
                boxes = np.stack([boxes[:, 0], h - boxes[:, 3], boxes[:, 2], h - boxes[:, 1]], 1)
        if self._chance("p_rot90"):
            for _ in range(int(self.rng.integers(1, 4))):
                img = np.rot90(img)   # counter-clockwise: (x, y) -> (y, w - x)
                if len(boxes):
                    boxes = np.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], 1)
                h, w = w, h
        return np.ascontiguousarray(img), boxes.reshape(-1, 4).astype(np.float32)

    def _affine(self, img: np.ndarray, boxes: np.ndarray, labels: List[str]
                ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        h, w = img.shape[:2]
        angle = self.rng.uniform(-self.p["rotate_deg"], self.p["rotate_deg"])
        scale = self.rng.uniform(*self.p["scale"])
        tx = self.rng.uniform(-self.p["translate"], self.p["translate"]) * w
        ty = self.rng.uniform(-self.p["translate"], self.p["translate"]) * h
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[:, 2] += (tx, ty)
        # constant fill (not reflect): reflection would clone objects without labels
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        if not len(boxes):
            return out, boxes, labels
        n = len(boxes)
        one = np.ones(n, np.float32)
        corners = np.stack([
            np.stack([boxes[:, 0], boxes[:, 1], one], 1),
            np.stack([boxes[:, 2], boxes[:, 1], one], 1),
            np.stack([boxes[:, 2], boxes[:, 3], one], 1),
            np.stack([boxes[:, 0], boxes[:, 3], one], 1)], 1)           # (n, 4, 3)
        pts = corners @ M.T.astype(np.float32)                           # (n, 4, 2)
        new = np.concatenate([pts.min(1), pts.max(1)], 1)                # axis-aligned envelope
        area_env = np.maximum((new[:, 2] - new[:, 0]) * (new[:, 3] - new[:, 1]), 1e-6)
        clipped = new.copy()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
        cw_, ch_ = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
        visible = (np.clip(cw_, 0, None) * np.clip(ch_, 0, None)) / area_env
        keep = (visible >= self.p["min_visible"]) & (cw_ >= cfg.MIN_BOX_PX) & (ch_ >= cfg.MIN_BOX_PX)
        return out, clipped[keep], [l for l, k in zip(labels, keep) if k]

    # ----- photometric
    def _brightness_contrast(self, img):
        a = 1.0 + self.rng.uniform(-self.p["contrast"], self.p["contrast"])
        b = self.rng.uniform(-self.p["brightness"], self.p["brightness"]) * 255.0
        return cv2.convertScaleAbs(img, alpha=a, beta=b)

    def _gamma(self, img):
        g = self.rng.uniform(*self.p["gamma"])
        lut = (np.power(np.linspace(0, 1, 256), 1.0 / g) * 255).astype(np.uint8)
        return cv2.LUT(img, lut)

    def _hsv(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = (hsv[..., 0].astype(np.int16) + int(self.rng.integers(-self.p["hue"], self.p["hue"] + 1))) % 180
        sat = hsv[..., 1].astype(np.float32) * self.rng.uniform(1 - self.p["sat"], 1 + self.p["sat"])
        val = hsv[..., 2].astype(np.float32) * self.rng.uniform(1 - self.p["val"], 1 + self.p["val"])
        out = np.stack([hue.astype(np.uint8), np.clip(sat, 0, 255).astype(np.uint8),
                        np.clip(val, 0, 255).astype(np.uint8)], -1)
        return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

    def _color_cast(self, img):          # LED colour temperature / liner tint
        gains = self.rng.uniform(1 - self.p["cast"], 1 + self.p["cast"], size=3).astype(np.float32)
        return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    def _gradient(self, img):            # uneven illumination across the plate
        h, w = img.shape[:2]
        yy, xx = self._grid(h, w)
        theta = self.rng.uniform(0, 2 * np.pi)
        proj = (xx - w / 2) * np.cos(theta) + (yy - h / 2) * np.sin(theta)
        proj /= (np.abs(proj).max() + 1e-6)
        gain = 1.0 + self.rng.uniform(0.10, 0.35) * proj
        return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)

    def _shadow(self, img):              # Dual sweeper/stopper arm shadow falling across the plate
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.float32)
        pts = self.rng.uniform([0, 0], [w, h], size=(int(self.rng.integers(3, 6)), 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        k = int(0.08 * max(h, w)) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        dark = self.rng.uniform(0.25, 0.50)
        return np.clip(img.astype(np.float32) * (1.0 - dark * mask)[..., None], 0, 255).astype(np.uint8)

    def _glare(self, img):               # Specular reflections on transparent/reflective vials & syringes
        h, w = img.shape[:2]
        yy, xx = self._grid(h, w)
        add = np.zeros((h, w), np.float32)
        for _ in range(int(self.rng.integers(1, 3))):
            cx, cy = self.rng.uniform(0, w), self.rng.uniform(0, h)
            r = self.rng.uniform(0.02, 0.08) * max(h, w)
            add += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        add *= self.rng.uniform(80, 200)
        return np.clip(img.astype(np.float32) + add[..., None], 0, 255).astype(np.uint8)

    def _motion_blur(self, img):         # item still moving when the frame is grabbed
        k = int(self.rng.choice([3, 5, 7, 9]))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D(((k - 1) / 2.0, (k - 1) / 2.0), self.rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        if kernel.sum() <= 0:
            return img
        return cv2.filter2D(img, -1, kernel / kernel.sum())

    def _gauss_blur(self, img):
        k = int(self.rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (k, k), 0)

    def _noise(self, img):
        sigma = self.rng.uniform(2, 8)
        noise = self.rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _jpeg(self, img):
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.rng.integers(40, 91))])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img

    def _clutter(self, img, boxes):
        """Opaque patches placed only where there is NO labelled object (liner folds, debris)."""
        h, w = img.shape[:2]
        img = img.copy()
        for _ in range(int(self.rng.integers(1, self.p["max_clutter"] + 1))):
            pw, ph = int(self.rng.uniform(0.04, 0.15) * w), int(self.rng.uniform(0.04, 0.15) * h)
            if pw < 4 or ph < 4 or pw >= w or ph >= h:
                continue
            x0, y0 = int(self.rng.integers(0, w - pw)), int(self.rng.integers(0, h - ph))
            if _overlap(np.array([x0, y0, x0 + pw, y0 + ph], np.float32), boxes) > 0.02:
                continue
            base = self.rng.integers(40, 220, size=3)
            patch = base + self.rng.normal(0, 12, (ph, pw, 3))
            img[y0:y0 + ph, x0:x0 + pw] = np.clip(patch, 0, 255).astype(np.uint8)
        return img

    # ----- public
    def __call__(self, img: np.ndarray, boxes: np.ndarray, labels: List[str],
                 require_objects: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Returns (image, boxes[xyxy float], labels) or None if every object got lost."""
        boxes = boxes.reshape(-1, 4).astype(np.float32)
        for _attempt in range(4):
            im, bx, lb = img, boxes, list(labels)
            im, bx = self._flip_rot90(im, bx)
            if self._chance("p_affine"):
                im, bx, lb = self._affine(im, bx, lb)
            if require_objects and len(boxes) and not len(bx):
                continue                                   # objects fell outside the frame; retry
            for key, fn in (("p_gradient", self._gradient), ("p_shadow", self._shadow),
                            ("p_glare", self._glare), ("p_brightness_contrast", self._brightness_contrast),
                            ("p_gamma", self._gamma), ("p_hsv", self._hsv), ("p_color_cast", self._color_cast),
                            ("p_motion_blur", self._motion_blur), ("p_gauss_blur", self._gauss_blur),
                            ("p_noise", self._noise)):
                if self._chance(key):
                    im = fn(im)
            if self._chance("p_clutter"):
                im = self._clutter(im, bx)
            if self._chance("p_jpeg"):
                im = self._jpeg(im)
            return im, bx, lb
        return None


# --------------------------------------------------------------------------- #
# Building the prepared dataset
# --------------------------------------------------------------------------- #
def _clean_boxes(boxes: np.ndarray, labels: Sequence[str], w: int, h: int) -> List[Box]:
    out: List[Box] = []
    for (x0, y0, x1, y1), name in zip(boxes, labels):
        x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
        x1, y1 = min(float(w), float(x1)), min(float(h), float(y1))
        if round(x1) - round(x0) >= 2 and round(y1) - round(y0) >= 2:
            out.append(Box(name, x0, y0, x1, y1))
    return out


def write_sample(out_dir: Path, stem: str, img: np.ndarray, boxes: Sequence[Box]) -> None:
    h, w = img.shape[:2]
    cv2.imwrite(str(out_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_voc_xml(out_dir / f"{stem}.xml", f"{stem}.jpg", w, h, boxes)


def copy_sanitised(sample: Sample, out_dir: Path) -> None:
    """Copy the original image untouched; rewrite XML with true size and clipped boxes."""
    dst = out_dir / sample.image_path.name
    shutil.copy2(str(sample.image_path), str(dst))
    write_voc_xml(out_dir / (sample.image_path.stem + ".xml"), dst.name, sample.width, sample.height, sample.boxes)


def build_prepared(all_samples: Dict[str, List[Sample]], augment: bool, bg_dir: Optional[Path], clean: bool) -> None:
    if clean and cfg.PREPARED_DIR.exists():
        shutil.rmtree(cfg.PREPARED_DIR)
    rng = np.random.default_rng(cfg.SEED)

    for split in ("val", "test"):
        out = cfg.PREPARED_SPLITS[split]
        out.mkdir(parents=True, exist_ok=True)
        for s in all_samples[split]:
            copy_sanitised(s, out)
        log.info("wrote %d sanitised %s images -> %s", len(all_samples[split]), split, out)

    train, out = all_samples["train"], cfg.PREPARED_SPLITS["train"]
    out.mkdir(parents=True, exist_ok=True)
    for s in train:
        copy_sanitised(s, out)
    log.info("wrote %d original train images -> %s", len(train), out)
    if not augment:
        return

    aug = WasteBinAugmenter(rng)
    bank = CropBank(train, cfg.CLASS_NAMES)
    log.info("crop bank: %d object crops", len(bank))
    copies = plan_copies(train, cfg.CLASS_NAMES, rng)
    n_aug = 0
    for s, n_copies in zip(train, copies):
        if n_copies == 0:
            continue
        img0 = imread(s.image_path)
        if img0 is None:
            continue
        boxes0 = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in s.boxes], np.float32).reshape(-1, 4)
        labels0 = [b.name for b in s.boxes]
        for k in range(n_copies):
            img, boxes, labels = img0, boxes0, labels0
            # Paste 1-3 extra objects to simulate mixed-waste scenes (up to 4 types together)
            if len(bank) and rng.random() < cfg.AUGMENTATION["p_paste_on_sample"]:
                img, boxes, labels = paste_crops(img, boxes, labels, bank, rng, int(rng.integers(1, 4)))
            res = aug(img, boxes, labels)
            if res is None:
                continue
            im, bx, lb = res
            write_sample(out, f"{s.image_path.stem}__aug{k}", im, _clean_boxes(bx, lb, im.shape[1], im.shape[0]))
            n_aug += 1
    log.info("wrote %d augmented train images (repeat-factor sampled)", n_aug)

    # Synthetic scenes on empty-bin backgrounds: breaks the object<->background correlation.
    bg_paths = sorted(p for p in bg_dir.glob("*") if p.suffix.lower() in IMG_EXTS) if bg_dir and bg_dir.is_dir() else []
    if bg_paths and len(bank) and cfg.N_COPY_PASTE_IMAGES > 0:
        n_syn = 0
        for i in range(cfg.N_COPY_PASTE_IMAGES):
            bg = imread(bg_paths[int(rng.integers(0, len(bg_paths)))])
            if bg is None:
                continue
            ref = train[int(rng.integers(0, len(train)))]
            bg = cv2.resize(bg, (ref.width, ref.height), interpolation=cv2.INTER_AREA)
            img, boxes, labels = paste_crops(bg, np.zeros((0, 4), np.float32), [], bank, rng, int(rng.integers(1, 5)))
            if not labels:
                continue
            res = aug(img, boxes, labels)
            if res is None:
                continue
            im, bx, lb = res
            write_sample(out, f"synthetic_{i:05d}__cp", im, _clean_boxes(bx, lb, im.shape[1], im.shape[0]))
            n_syn += 1
        log.info("wrote %d copy-paste images on %d background photos", n_syn, len(bg_paths))
    else:
        log.info("no backgrounds in %s -> skipping copy-paste synthesis", bg_dir)

    samples, _, _ = load_split(out, cfg.CLASS_NAMES)
    stats = class_statistics(samples, cfg.CLASS_NAMES)
    log.info("prepared train: %d images, imbalance %.1f:1 (was %.1f:1) | BMW Bins: %s",
             len(samples), imbalance_ratio(stats),
             imbalance_ratio(class_statistics(train, cfg.CLASS_NAMES)),
             bmw_bin_statistics(samples))
    for c, v in stats.items():
        log.info("    %-12s %5d objects", c, v["instances"])


# --------------------------------------------------------------------------- #
# Model Maker loading (training env only)
# --------------------------------------------------------------------------- #
def make_loader(split_dir: Path, class_names: Sequence[str] = cfg.CLASS_NAMES):
    """tflite-model-maker DataLoader for one prepared split (images and XML share a folder)."""
    from tflite_model_maker import object_detector  # lazy: only exists in the training env
    return object_detector.DataLoader.from_pascal_voc(str(split_dir), str(split_dir), list(class_names))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover")
    sub.add_parser("verify")
    b = sub.add_parser("build")
    b.add_argument("--no-augment", action="store_true")
    b.add_argument("--bg-dir", type=Path, default=cfg.BACKGROUND_DIR)
    b.add_argument("--clean", action="store_true", help="delete data/prepared first")
    args = ap.parse_args(argv)

    if args.cmd == "discover":
        found = discover_classes(cfg.RAW_SPLITS)
        print("Class names found in XML:", dict(found))
        print("config.CLASS_NAMES      :", cfg.CLASS_NAMES)
        print("BMW 2016 Bin Mapping    :", cfg.CLASS_TO_BIN)
        return 0

    all_samples, report, errors = verify_dataset(cfg.RAW_SPLITS, cfg.CLASS_NAMES)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORT_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2))
    if errors:
        for e in errors:
            log.error(e)
        log.error("Fix the errors above (hint: `python dataset_prep.py discover`).")
        return 1
    if args.cmd == "build":
        build_prepared(all_samples, augment=not args.no_augment, bg_dir=args.bg_dir, clean=args.clean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
