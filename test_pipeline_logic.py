#!/usr/bin/env python3
"""Hardware-free tests (NumPy + OpenCV + SQLite only; no TensorFlow, no camera, no GPIO)
for the Smart Mobile Medical-Waste Collection and Segregation System (SIH 2026 - Team Pravah77).

    python test_pipeline_logic.py          # or: pytest -q test_pipeline_logic.py
"""
from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import cv2
import numpy as np

import config as cfg
import dataset_prep as dp
import edge_inference as ei
import flap_config as fcfg
import metrics
import tflite_utils as tu


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def _scene(w=200, h=160, box=(30, 20, 90, 70)):
    img = np.full((h, w, 3), 40, np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = (0, 255, 255)
    return img, np.array([box], np.float32)


def _measured_box(img):
    mask = (img[..., 1] > 200) & (img[..., 2] > 200) & (img[..., 0] < 60)
    ys, xs = np.where(mask)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _only(**on):
    off = {k: 0.0 for k in cfg.AUGMENTATION if k.startswith("p_")}
    off.update(on)
    return off


# --------------------------------------------------------------------------- #
def test_flips_and_rot90_keep_boxes_on_the_object():
    for params in (_only(p_hflip=1.0), _only(p_vflip=1.0), _only(p_rot90=1.0), _only(p_hflip=1.0, p_vflip=1.0, p_rot90=1.0)):
        for seed in range(6):
            aug = dp.WasteBinAugmenter(np.random.default_rng(seed), params)
            img, boxes = _scene()
            out, bx, lb = aug(img, boxes, ["a"])
            assert _box_iou(bx[0], _measured_box(out)) > 0.97, (params, seed, bx[0], _measured_box(out))


def test_affine_boxes_cover_the_object():
    for seed in range(20):
        aug = dp.WasteBinAugmenter(np.random.default_rng(seed), _only(p_affine=1.0))
        img, boxes = _scene()
        res = aug(img, boxes, ["a"])
        assert res is not None
        out, bx, lb = res
        assert len(bx) == 1 and lb == ["a"]
        m = _measured_box(out)
        assert bx[0][0] <= m[0] + 2 and bx[0][1] <= m[1] + 2 and bx[0][2] >= m[2] - 2 and bx[0][3] >= m[3] - 2
        assert _box_iou(bx[0], m) > 0.55
        assert bx.min() >= 0 and bx[:, 2].max() <= out.shape[1] and bx[:, 3].max() <= out.shape[0]


def test_full_augmentation_never_crashes_and_stays_valid():
    rng = np.random.default_rng(0)
    aug = dp.WasteBinAugmenter(rng, {k: 1.0 for k in cfg.AUGMENTATION if k.startswith("p_")})
    for _ in range(15):
        img, boxes = _scene()
        res = aug(img, boxes, ["a"])
        if res is None:
            continue
        out, bx, lb = res
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
        assert len(bx) == len(lb)
        assert np.all(bx[:, 2] > bx[:, 0]) and np.all(bx[:, 3] > bx[:, 1])


def test_copy_paste_adds_exact_boxes_without_overlap():
    img, _ = _scene(400, 400, (50, 50, 150, 150))
    bank = dp.CropBank.__new__(dp.CropBank)
    bank.crops, bank.names, bank.p = [img[40:160, 40:160].copy()], ["a"], np.array([1.0])
    rng = np.random.default_rng(1)
    _, bx, lb = dp.paste_crops(np.full((400, 400, 3), 90, np.uint8), np.zeros((0, 4), np.float32), [], bank, rng, 3)
    assert len(bx) == len(lb) >= 1
    for i in range(len(bx)):
        for j in range(i + 1, len(bx)):
            assert metrics.iou_matrix(bx[i:i + 1], bx[j:j + 1])[0, 0] < 0.2


def test_repeat_factors_favour_rare_classes():
    mk = lambda names: dp.Sample(Path("a"), Path("a"), 10, 10, [dp.Box(n, 0, 0, 5, 5) for n in names])
    samples = [mk(["common"]) for _ in range(90)] + [mk(["rare"]) for _ in range(5)] + [mk(["common", "rare"])] * 5
    f, r = dp.repeat_factors(samples, ["common", "rare"], 0.3)
    assert r["rare"] > r["common"] >= 1.0
    assert f[0] == r["common"] and f[90] == r["rare"]


# --------------------------------------------------------------------------- #
def test_voc_roundtrip_clip_unknown_and_degenerate():
    tmp = Path(tempfile.mkdtemp())
    try:
        cv2.imwrite(str(tmp / "a.jpg"), np.zeros((100, 120, 3), np.uint8))
        boxes = [dp.Box("mask", -5, 10, 60, 50), dp.Box("alien", 1, 1, 30, 30),
                 dp.Box("vial", 10, 10, 11, 11), dp.Box("sharps", 100, 80, 200, 200)]
        dp.write_voc_xml(tmp / "a.xml", "a.jpg", 999, 999, boxes)
        samples, issues, unknown = dp.load_split(tmp, cfg.CLASS_NAMES)
        s = samples[0]
        assert (s.width, s.height) == (120, 100)
        assert unknown == {"alien": 1}
        assert len(issues["degenerate_box"]) == 1
        names = {b.name: b for b in s.boxes}
        assert names["mask"].xmin == 0 and names["sharps"].xmax == 120 and names["sharps"].ymax == 100
    finally:
        shutil.rmtree(tmp)


def test_build_pipeline_end_to_end_and_leakage_detection():
    tmp = Path(tempfile.mkdtemp())
    old = (cfg.PREPARED_DIR, dict(cfg.PREPARED_SPLITS), cfg.REPORT_DIR, cfg.N_COPY_PASTE_IMAGES)
    try:
        raw = {k: tmp / "raw" / k for k in ("train", "val", "test")}
        rng = np.random.default_rng(0)
        classes = cfg.CLASS_NAMES
        for split, n in (("train", 24), ("val", 6), ("test", 6)):
            raw[split].mkdir(parents=True)
            for i in range(n):
                img = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
                cls = classes[0] if i % 6 else classes[3]
                x0, y0 = int(rng.integers(5, 60)), int(rng.integers(5, 60))
                img[y0:y0 + 50, x0:x0 + 50] = 200
                name = f"{split}_{i}_jpg.rf.{i:04d}"
                cv2.imwrite(str(raw[split] / f"{name}.jpg"), img)
                dp.write_voc_xml(raw[split] / f"{name}.xml", f"{name}.jpg", 160, 160, [dp.Box(cls, x0, y0, x0 + 50, y0 + 50)])
        shutil.copy(raw["train"] / "train_0_jpg.rf.0000.jpg", raw["val"] / "val_leak_jpg.rf.9999.jpg")
        dp.write_voc_xml(raw["val"] / "val_leak_jpg.rf.9999.xml", "val_leak_jpg.rf.9999.xml", 160, 160, [dp.Box(classes[3], 5, 5, 55, 55)])

        samples, report, errors = dp.verify_dataset(raw, classes)
        assert not errors, errors
        assert report["leakage"], "planted near-duplicate was not detected"
        assert "bmw_bin_stats" in report["splits"]["train"]

        cfg.PREPARED_DIR = tmp / "prepared"
        cfg.PREPARED_SPLITS.update({k: cfg.PREPARED_DIR / k for k in ("train", "val", "test")})
        cfg.N_COPY_PASTE_IMAGES = 5
        bgdir = tmp / "bg"
        bgdir.mkdir()
        cv2.imwrite(str(bgdir / "empty.jpg"), np.full((120, 200, 3), 80, np.uint8))
        dp.build_prepared(samples, augment=True, bg_dir=bgdir, clean=True)

        out, _, _ = dp.load_split(cfg.PREPARED_SPLITS["train"], classes)
        assert len(out) > len(samples["train"]), "no augmented images were written"
        assert any("__cp" in s.image_path.stem for s in out), "no copy-paste images"
        stats_before = dp.class_statistics(samples["train"], classes)
        stats_after = dp.class_statistics(out, classes)
        ratio = lambda st: st[classes[0]]["instances"] / max(st[classes[3]]["instances"], 1)
        assert ratio(stats_after) < ratio(stats_before), "repeat-factor sampling did not reduce imbalance"
        for s in out:
            im = dp.imread(s.image_path)
            assert (s.width, s.height) == (im.shape[1], im.shape[0])
            for b in s.boxes:
                assert 0 <= b.xmin < b.xmax <= s.width and 0 <= b.ymin < b.ymax <= s.height
    finally:
        cfg.PREPARED_DIR, restored, cfg.REPORT_DIR, cfg.N_COPY_PASTE_IMAGES = old
        cfg.PREPARED_SPLITS.clear(); cfg.PREPARED_SPLITS.update(restored)
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
class _FakeInterp:
    """Mimics a Model Maker EfficientDet-Lite TFLite interpreter, with SHUFFLED output order."""

    def __init__(self, order=(0, 1, 2, 3), quantised=False):
        self.order, self.quantised = order, quantised
        self.fed = None

    def get_input_details(self):
        return [{"index": 0, "dtype": np.uint8, "shape": np.array([1, 320, 320, 3])}]

    def _raw(self):
        count = np.array([2.0], np.float32)
        scores = np.zeros((1, 25), np.float32); scores[0, :2] = (0.91, 0.55)
        classes = np.zeros((1, 25), np.float32); classes[0, :2] = (2, 0)
        boxes = np.zeros((1, 25, 4), np.float32)
        boxes[0, 0] = (0.25, 0.10, 0.75, 0.50)
        boxes[0, 1] = (0.00, 0.60, 0.20, 1.00)
        return [count, scores, classes, boxes]

    def get_output_details(self):
        return [{"index": i, "dtype": a.dtype.type, "shape": np.array(a.shape), "quantization": (0.0, 0)}
                for i, a in enumerate(self._permuted())]

    def _permuted(self):
        raw = self._raw()
        return [raw[i] for i in self.order]

    def set_tensor(self, idx, x):
        self.fed = x.copy()

    def invoke(self):
        pass

    def get_tensor(self, idx):
        return self._permuted()[idx].copy()


def _detector(monkeypatch_order=(0, 1, 2, 3)):
    tu.load_interpreter = lambda *a, **k: _FakeInterp(monkeypatch_order)
    return tu.TFLiteDetector(Path("fake.tflite"))


def test_output_tensor_order_does_not_matter_and_boxes_scale_to_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 2] = 255
    for order in [(0, 1, 2, 3), (1, 3, 0, 2), (3, 2, 1, 0), (2, 0, 3, 1)]:
        det = _detector(order)
        dets, _ = det.infer(frame, 0.3)
        assert [d.class_id for d in dets] == [2, 0], order
        x0, y0, x1, y1 = dets[0].box
        assert (round(x0), round(y0), round(x1), round(y1)) == (64, 120, 320, 360), dets[0].box
        assert det.interp.fed.shape == (1, 320, 320, 3)
        assert det.interp.fed[0, 0, 0].tolist() == [255, 0, 0]


def test_roi_offsets_are_restored():
    det = _detector()
    frame = np.zeros((480, 640, 3), np.uint8)
    dets, _ = det.infer(frame, 0.3, roi=(100, 50, 320, 240))
    x0, y0, x1, y1 = dets[0].box
    assert (round(x0), round(y0)) == (100 + 32, 50 + 60)


def test_nms_class_aware_vs_agnostic():
    a = tu.Detection(0, 0.9, (10, 10, 100, 100))
    b = tu.Detection(1, 0.8, (12, 12, 98, 98))
    c = tu.Detection(0, 0.7, (14, 14, 99, 99))
    assert len(tu.nms([a, b, c], 0.5, class_agnostic=False)) == 2
    assert [d.class_id for d in tu.nms([a, b, c], 0.5, class_agnostic=True)] == [0]
    far = tu.Detection(0, 0.6, (300, 300, 380, 380))
    assert len(ei.postprocess([a, b, c, far])) == 2


# --------------------------------------------------------------------------- #
def test_stable_trigger_requires_streak_cooldown_and_priority():
    names = ["gloves", "mask", "sharps", "syringe", "vial"]
    g = ei.StableTrigger(names, 0.7, 3, 2.0, cfg.CLASS_PRIORITY)
    D = tu.Detection
    t = 0.0
    assert g.update([D(0, 0.9, (0, 0, 1, 1))], t) is None
    assert g.update([D(0, 0.9, (0, 0, 1, 1))], t + .1) is None
    assert g.update([D(0, 0.5, (0, 0, 1, 1))], t + .2) is None
    assert g.update([D(0, 0.9, (0, 0, 1, 1))], t + .3) is None
    assert g.update([D(0, 0.9, (0, 0, 1, 1))], t + .4) is None
    ev = g.update([D(0, 0.9, (0, 0, 1, 1))], t + .5)
    assert ev and ev["class_id"] == 0 and not ev["mixed"]
    for k in range(5):
        assert g.update([D(0, 0.95, (0, 0, 1, 1))], t + 1.0 + k * .1) is None
    # after cooldown: sharps (priority) beats gloves when both are stable; mixed flag & boxes set
    both = [D(0, 0.9, (10, 10, 50, 50)), D(2, 0.8, (200, 10, 260, 50))]
    ev = None
    for k in range(4):
        ev = g.update(both, t + 3.0 + k * .1) or ev
    assert ev and ev["class_id"] == 2 and ev["mixed"]
    assert len(ev["secondary_boxes"]) == 1


def test_sorting_executor_is_non_blocking_and_drops_while_busy():
    import threading, time
    gate, calls = threading.Event(), []
    orig = ei.trigger_sorting_mechanism
    ei.trigger_sorting_mechanism = lambda cid, conf, **ctx: (gate.wait(2), calls.append(cid))
    try:
        ex = ei.SortingExecutor()
        assert ex.submit(1, 0.9) is True
        assert ex.submit(2, 0.9) is False
        gate.set()
        time.sleep(0.2)
        assert calls == [1] and not ex.busy
    finally:
        ei.trigger_sorting_mechanism = orig


# --------------------------------------------------------------------------- #
# SIH 2026 New Requirements Tests: 4-Bin Mapping, Flap Config, Fail-Safe, DB
# --------------------------------------------------------------------------- #
def test_bmw_2016_four_bin_mapping_and_flap_profiles():
    assert set(cfg.BMW_BINS) == {"A", "B", "C", "D"}
    assert set(fcfg.BIN_PROFILES.keys()) == {"A", "B", "C", "D"}
    # Verify all 4 bins (Yellow, Red, White, Blue) are represented in CLASS_TO_BIN
    mapped_bins = set(cfg.CLASS_TO_BIN.values())
    assert mapped_bins == {"A", "B", "C", "D"}, f"Expected all 4 bins A/B/C/D, got {mapped_bins}"

    # Verify separation line angle calculation for mixed waste on the dished plate
    sep_deg = fcfg.compute_separation_line_angle((100, 200, 150, 250), [(300, 200, 350, 250)])
    assert abs(abs(sep_deg) - 90.0) < 1e-3  # horizontally separated -> vertical separation line (90 deg)

    plan_mixed = fcfg.build_flap_action_plan(
        bin_code="C",
        mixed_waste=True,
        primary_box=(100, 200, 150, 250),
        secondary_boxes=[(300, 200, 350, 250)],
    )
    assert plan_mixed.target_bin == "C"
    assert plan_mixed.engage_stopper is True
    assert plan_mixed.stopper_angle_deg == fcfg.BIN_PROFILES["C"].stopper_hold_deg


def test_highest_hazard_fail_safe_and_sensor_fusion():
    # 1. High confidence normal sort -> Bin D (Blue - Glassware) for vial
    b, fs, reason = ei.resolve_bin_with_fail_safe("vial", 0.88)
    assert (b, fs, reason) == ("D", False, "cnn_confident_sort")

    # 2. Composite syringe + sharps -> Highest Hazard Fail-Safe overrides to Bin C (White - Sharps)
    b, fs, reason = ei.resolve_bin_with_fail_safe("syringe", 0.92, mixed=True, co_detected_classes=["syringe", "sharps"])
    assert (b, fs, reason) == ("C", True, "composite_syringe_with_sharp_fail_safe")

    # 3. Low camera confidence + Load Cell & Color Sensor recovery -> Bin A (Yellow)
    b, fs, reason = ei.resolve_bin_with_fail_safe("unknown", 0.35, weight_g=15.0, color_cue="yellow")
    assert (b, fs, reason) == ("A", True, "sensor_fusion_color_loadcell_recovery")

    # 4. Ambiguous composite item without colour cue -> Highest Hazard Fail-Safe (Bin C)
    b, fs, reason = ei.resolve_bin_with_fail_safe("unknown", 0.40, weight_g=15.0, color_cue=None)
    assert (b, fs, reason) == (cfg.FALLBACK_BIN, True, "highest_hazard_fail_safe")


def test_trigger_sorting_mechanism_and_sqlite_traceability():
    tmp = Path(tempfile.mkdtemp())
    old_sqlite = cfg.SQLITE_FALLBACK_PATH
    try:
        cfg.SQLITE_FALLBACK_PATH = tmp / "trace.sqlite3"
        ei.init_sorting_mechanism(gpio_enabled=False)
        ei._trace_db = ei.WasteTraceabilityDB(postgres_dsn="postgresql://invalid:5432/none", sqlite_path=cfg.SQLITE_FALLBACK_PATH)

        # Sort a sharp (id 2 -> Bin C) and a mixed glove+vial dump
        assert ei.trigger_sorting_mechanism(2, 0.93, mixed=False, weight_g=4.2) is True
        assert ei.trigger_sorting_mechanism(
            4, 0.89, mixed=True, weight_g=28.5,
            primary_box=(50, 50, 120, 120), secondary_boxes=[(250, 50, 310, 120)],
        ) is True
        # Unknown id -> fallback highest-hazard bin C
        assert ei.trigger_sorting_mechanism(99, 0.50, weight_g=10.0) is True

        # Emergency stop halts sorting immediately
        ei._sensors.trigger_emergency_stop()
        assert ei.trigger_sorting_mechanism(0, 0.95) is False
        ei._sensors.clear_emergency_stop()

        with sqlite3.connect(str(cfg.SQLITE_FALLBACK_PATH)) as conn:
            rows = conn.execute("SELECT class_name, bmw_bin, mixed_waste, fail_safe FROM waste_events ORDER BY id").fetchall()
        assert len(rows) == 3
        assert rows[0] == ("sharps", "C", 0, 0)
        assert rows[1] == ("vial", "D", 1, 0)
        assert rows[2] == ("unknown", "C", 0, 1)
    finally:
        cfg.SQLITE_FALLBACK_PATH = old_sqlite
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
def test_per_class_precision_recall_and_confusion():
    names = ["a", "b"]
    S = lambda *bs: dp.Sample(Path("x"), Path("x"), 100, 100, [dp.Box(n, *c) for n, c in bs])
    D = tu.Detection
    samples = [S(("a", (10, 10, 50, 50)), ("b", (60, 60, 90, 90))), S(("a", (10, 10, 50, 50)))]
    preds = [[D(0, 0.9, (10, 10, 50, 50)),
              D(0, 0.8, (60, 60, 90, 90))],
             [D(1, 0.95, (70, 70, 99, 99))]]
    r = metrics.score_predictions(samples, preds, names, 0.5)
    pa, pb = r["per_class"]["a"], r["per_class"]["b"]
    assert (pa["tp"], pa["fp"], pa["fn"]) == (1, 1, 1)
    assert (pb["tp"], pb["fp"], pb["fn"]) == (0, 1, 1)
    assert r["confusion"]["b"]["a"] == 1 and r["confusion"]["background"]["b"] == 1 and r["confusion"]["a"]["missed"] == 1
    assert abs(pa["precision"] - 0.5) < 1e-9 and abs(pa["recall"] - 0.5) < 1e-9


if __name__ == "__main__":
    import logging, traceback
    logging.disable(logging.CRITICAL)
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:  # noqa: BLE001
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    raise SystemExit(1 if fails else 0)
