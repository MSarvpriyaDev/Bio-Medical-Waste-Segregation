"""TFLite detector wrapper shared by the Raspberry Pi 5 runtime and host-side evaluation.

Deliberately has NO hard TensorFlow dependency: it prefers ``ai_edge_litert``,
then ``tflite_runtime`` (what you install on the Pi), and only falls back to
``tensorflow.lite`` on a training host.

Includes software specular-glare suppression (supporting the physical polarizing
filter on the camera) to eliminate optical distortion on transparent and
reflective surfaces (glass vials, plastic syringes, wet bio-fluid surfaces).

Model Maker EfficientDet-Lite output convention (4 float tensors, NMS already
applied inside the graph):
    boxes   [1, N, 4]  (ymin, xmin, ymax, xmax), normalised 0..1
    classes [1, N]     0-based index into labels.txt
    scores  [1, N]
    count   [1]
The order of those tensors in ``get_output_details()`` is NOT guaranteed, so they
are identified by shape and value instead of position.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("tflite_utils")


@dataclass
class Detection:
    class_id: int
    score: float
    box: Tuple[float, float, float, float]   # x0, y0, x1, y1 in source-frame pixels

    @property
    def centroid(self) -> Tuple[float, float]:
        return (0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3]))

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def suppress_specular_glare(frame_bgr: np.ndarray, v_thresh: int = 245, s_thresh: int = 30) -> np.ndarray:
    """Clamps extreme specular reflections on transparent/reflective vials & syringes
    while preserving colour cues on non-saturated pixels.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = (hsv[..., 2] >= v_thresh) & (hsv[..., 1] <= s_thresh)
    if not np.any(glare_mask):
        return frame_bgr
    hsv = hsv.copy()
    hsv[glare_mask, 2] = 210
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def load_interpreter(model_path: Path, num_threads: int = 4):
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            import tensorflow as tf  # training host only
            Interpreter = tf.lite.Interpreter  # type: ignore
    interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interp.allocate_tensors()
    return interp


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def _looks_integral(v: np.ndarray) -> bool:
    return bool(v.size) and bool(np.all(np.abs(v - np.round(v)) < 1e-6))


def assign_outputs(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]:
    """Identify (boxes[N,4], scores[N], classes[N], count) from the raw output tensors."""
    boxes: Optional[np.ndarray] = None
    count: Optional[int] = None
    rest: List[np.ndarray] = []
    for a in arrays:
        if a.ndim == 3 and a.shape[-1] == 4:
            boxes = a.reshape(-1, 4)
        elif a.size == 1:
            count = int(round(float(a.reshape(-1)[0])))
        else:
            rest.append(a.reshape(-1))
    if boxes is None or len(rest) != 2:
        raise RuntimeError("Unexpected detector outputs: %s" % [tuple(a.shape) for a in arrays])
    n = count if count is not None else len(rest[0])
    a, b = rest
    head = max(min(n, len(a)), 1)
    if _looks_integral(a[:head]) and not _looks_integral(b[:head]):
        classes, scores = a, b
    elif _looks_integral(b[:head]) and not _looks_integral(a[:head]):
        classes, scores = b, a
    else:                       # ambiguous only when there is nothing to report anyway
        scores, classes = a, b
    return boxes, scores, classes, count


def decode_detections(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, count: Optional[int],
                      frame_w: int, frame_h: int, in_w: int, in_h: int, score_thr: float) -> List[Detection]:
    n = len(scores) if count is None else min(count, len(scores))
    n = min(n, len(boxes))
    if n <= 0:
        return []
    b, s, c = boxes[:n], scores[:n], classes[:n]
    keep = s >= score_thr
    if not keep.any():
        return []
    b, s, c = b[keep], s[keep], c[keep]
    if b.size and float(b.max()) > 1.5:             # defensive: pixel coords instead of normalised
        b = b / np.array([in_h, in_w, in_h, in_w], np.float32)
    b = np.clip(b, 0.0, 1.0)
    out: List[Detection] = []
    for (ymin, xmin, ymax, xmax), sc, cl in zip(b, s, c):
        x0, x1 = xmin * frame_w, xmax * frame_w
        y0, y1 = ymin * frame_h, ymax * frame_h
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        out.append(Detection(int(round(float(cl))), float(sc), (float(x0), float(y0), float(x1), float(y1))))
    return out


def nms(dets: Sequence[Detection], iou_thr: float, class_agnostic: bool = False) -> List[Detection]:
    """Greedy NMS in NumPy. class_agnostic=True also merges boxes of different classes."""
    if len(dets) < 2:
        return list(dets)
    boxes = np.array([d.box for d in dets], np.float32)
    scores = np.array([d.score for d in dets], np.float32)
    if not class_agnostic:      # offset trick: boxes of different classes can never overlap
        cls = np.array([d.class_id for d in dets], np.float32)
        boxes = boxes + (cls * (float(boxes.max()) + 1.0))[:, None]
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0) * (y1 - y0)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        ih = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return [dets[i] for i in keep]


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class TFLiteDetector:
    def __init__(self, model_path: Path, num_threads: int = 4, glare_filter: bool = False):
        self.model_path = Path(model_path)
        self.glare_filter = glare_filter
        self.interp = load_interpreter(self.model_path, num_threads)
        d = self.interp.get_input_details()[0]
        self._in_index = d["index"]
        self._in_dtype = d["dtype"]
        self.in_h, self.in_w = int(d["shape"][1]), int(d["shape"][2])
        if self._in_dtype not in (np.uint8, np.int8):
            raise NotImplementedError(
                "Input dtype %s: Model Maker EfficientDet-Lite models take uint8 RGB (0-255). "
                "Re-export with export_tflite.py." % self._in_dtype)
        self._out_details = self.interp.get_output_details()
        self._buf = np.zeros((1, self.in_h, self.in_w, 3), dtype=self._in_dtype)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Stretch-resize (what the TFLite Task Library and Model Maker's own demo do);
        # resize first, THEN apply glare filter / convert colour on 320x320 not 640x480.
        small = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        if self.glare_filter:
            small = suppress_specular_glare(small)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self._in_dtype == np.int8:
            rgb = (rgb.astype(np.int16) - 128).astype(np.int8)
        self._buf[0] = rgb
        return self._buf

    def _read_outputs(self):
        arrays = []
        for d in self._out_details:
            a = self.interp.get_tensor(d["index"])
            if a.dtype.kind in "iu":                       # quantised output -> real values
                scale, zero = d["quantization"]
                a = (a.astype(np.float32) - zero) * (scale if scale else 1.0)
            arrays.append(np.asarray(a, dtype=np.float32))
        return assign_outputs(arrays)

    def infer(self, frame_bgr: np.ndarray, score_threshold: float = 0.3,
              roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[List[Detection], Dict[str, float]]:
        """Run one frame. Boxes are returned in ORIGINAL frame pixels (ROI offset re-applied)."""
        ox = oy = 0
        if roi is not None:
            ox, oy, rw, rh = roi
            frame_bgr = frame_bgr[oy:oy + rh, ox:ox + rw]
        h0, w0 = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        x = self._preprocess(frame_bgr)
        self.interp.set_tensor(self._in_index, x)
        t1 = time.perf_counter()
        self.interp.invoke()
        t2 = time.perf_counter()
        boxes, scores, classes, count = self._read_outputs()
        dets = decode_detections(boxes, scores, classes, count, w0, h0, self.in_w, self.in_h, score_threshold)
        if ox or oy:
            dets = [Detection(d.class_id, d.score, (d.box[0] + ox, d.box[1] + oy, d.box[2] + ox, d.box[3] + oy))
                    for d in dets]
        t3 = time.perf_counter()
        return dets, {"pre_ms": (t1 - t0) * 1e3, "invoke_ms": (t2 - t1) * 1e3,
                      "post_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}


def benchmark(model_path: Path, runs: int = 50, warmup: int = 5, num_threads: int = 4) -> Dict[str, float]:
    """Latency on THIS machine. Run it on the Raspberry Pi 5 for numbers that matter."""
    det = TFLiteDetector(model_path, num_threads)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(warmup):
        det.infer(frame, 1.1)
    inv, tot = [], []
    for _ in range(runs):
        _, t = det.infer(frame, 1.1)
        inv.append(t["invoke_ms"])
        tot.append(t["total_ms"])
    return {"threads": num_threads, "invoke_mean_ms": float(np.mean(inv)),
            "invoke_p50_ms": float(np.percentile(inv, 50)), "invoke_p95_ms": float(np.percentile(inv, 95)),
            "pipeline_mean_ms": float(np.mean(tot)), "fps_ceiling": 1000.0 / float(np.mean(tot))}
