#!/usr/bin/env python3
"""Export the trained EfficientDet-Lite0 to TFLite, quantise, verify, and package
for the Smart Mobile Medical-Waste Collection and Segregation System
(SIH 2026 - PS ID 26115 - Team Pravah77).

Variants
--------
float32  reference (accuracy baseline, ~13 MB)
float16  weights stored as fp16 (~half size). On a Raspberry Pi CPU this saves SIZE,
         not time: weights are expanded to fp32 at load.
int8     full-integer quantisation, calibrated on a representative dataset. This is
         the variant that is fastest on Raspberry Pi 5 (Cortex-A76) and OpenMV.

Representative dataset
----------------------
tflite-model-maker builds the calibration generator itself from a DataLoader.
``build_calibration_subset`` therefore creates a small, class-stratified folder of
ORIGINAL (un-augmented) training frames - the same distribution the camera will
see - and ``QuantizationConfig.for_int8`` pulls ``CALIBRATION_IMAGES`` samples
from it. Poor or too-uniform calibration data is the usual reason INT8 loses mAP.

Normally this module is driven by train.py right after training. Standalone use
(experimental checkpoint restore, see ``restore_model``):

    python export_tflite.py --from-checkpoint --variants float32 float16 int8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import config as cfg
import flap_config as fcfg
from dataset_prep import class_statistics, load_split, make_loader
from tflite_utils import TFLiteDetector, benchmark

log = logging.getLogger("export_tflite")


# --------------------------------------------------------------------------- #
# Representative dataset
# --------------------------------------------------------------------------- #
def build_calibration_subset(n: int = cfg.CALIBRATION_IMAGES, seed: int = cfg.SEED) -> Path:
    """Class-stratified random sample of original training frames (round-robin over rarest class)."""
    samples, _, _ = load_split(cfg.PREPARED_SPLITS["train"], cfg.CLASS_NAMES)
    originals = [s for s in samples if s.boxes and "__aug" not in s.image_path.stem and "__cp" not in s.image_path.stem]
    if not originals:
        raise RuntimeError("No original training images found in %s" % cfg.PREPARED_SPLITS["train"])
    stats = class_statistics(originals, cfg.CLASS_NAMES)
    buckets: Dict[str, list] = defaultdict(list)
    for s in originals:
        buckets[min((b.name for b in s.boxes), key=lambda c: stats[c]["instances"])].append(s)
    rng = random.Random(seed)
    for items in buckets.values():
        rng.shuffle(items)
    chosen, order = [], sorted(buckets)
    while len(chosen) < n and any(buckets.values()):
        for c in order:
            if buckets[c] and len(chosen) < n:
                chosen.append(buckets[c].pop())
    dst = cfg.CALIBRATION_DIR
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for s in chosen:
        shutil.copy2(str(s.image_path), str(dst / s.image_path.name))
        shutil.copy2(str(s.xml_path), str(dst / s.xml_path.name))
    log.info("calibration subset: %d images -> %s", len(chosen), dst)
    return dst


def build_calibration_loader():
    return make_loader(build_calibration_subset())


# --------------------------------------------------------------------------- #
# Quantisation configs
# --------------------------------------------------------------------------- #
def quantization_candidates(variant: str, calib_loader) -> list:
    """Configs to try in order; the first that converts wins."""
    from tflite_model_maker.config import QuantizationConfig  # lazy: training env only
    if variant == "float32":
        return [QuantizationConfig()]               # empty config == no quantisation
    if variant == "float16":
        return [QuantizationConfig.for_float16()]
    if variant == "dynamic_range":
        return [QuantizationConfig.for_dynamic()]
    if variant == "int8":
        steps = min(cfg.CALIBRATION_IMAGES, calib_loader.size)
        cands = []
        try:    # uint8 image in, float detections out (what edge_inference.py expects)
            import tensorflow as tf
            cands.append(QuantizationConfig.for_int8(calib_loader, quantization_steps=steps,
                                                     inference_input_type=tf.uint8,
                                                     inference_output_type=tf.float32))
        except TypeError:
            pass
        cands.append(QuantizationConfig.for_int8(calib_loader, quantization_steps=steps))
        return cands
    raise ValueError("unknown variant %r" % variant)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_tflite(path: Path, num_threads: int = cfg.NUM_THREADS, runs: int = 30) -> dict:
    size_mb = path.stat().st_size / 1e6
    det = TFLiteDetector(path, num_threads)       # raises if the I/O contract is not what the Pi 5 code expects
    io = {"input_shape": [1, det.in_h, det.in_w, 3], "input_dtype": str(det._in_dtype.__name__),
          "outputs": [{"shape": [int(x) for x in d["shape"]], "dtype": str(d["dtype"].__name__)} for d in det._out_details]}
    lat = benchmark(path, runs=runs, num_threads=num_threads)
    if size_mb > cfg.MODEL_SIZE_BUDGET_MB:
        log.warning("%s is %.1f MB (> budget %.1f MB)", path.name, size_mb, cfg.MODEL_SIZE_BUDGET_MB)
    log.info("%s: %.2f MB | host latency %.1f ms (p95 %.1f) -> ceiling %.1f FPS "
             "[HOST numbers - benchmark on the Pi 5: `python edge_inference.py --benchmark`]",
             path.name, size_mb, lat["invoke_mean_ms"], lat["invoke_p95_ms"], lat["fps_ceiling"])
    return {"size_mb": size_mb, "io": io, "host_latency": lat}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Export pipeline
# --------------------------------------------------------------------------- #
def export_variants(model, variants: Sequence[str], calib_loader, test_loader, test_dir: Path,
                    export_dir: Path = cfg.EXPORT_DIR) -> Dict[str, dict]:
    from tflite_model_maker.config import ExportFormat  # lazy
    import metrics

    results: Dict[str, dict] = {}
    for v in variants:
        out_dir = export_dir / v
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = "%s_%s.tflite" % (cfg.MODEL_NAME, v)
        path = out_dir / fname
        error: Optional[str] = None
        for qc in quantization_candidates(v, calib_loader):
            try:
                model.export(export_dir=str(out_dir), tflite_filename=fname, label_filename=cfg.DEPLOY_LABELS_NAME,
                             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL], quantization_config=qc)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - try the next config, report if all fail
                error = "%s: %s" % (type(exc).__name__, exc)
                log.warning("export of %s failed with one config (%s)", v, error)
        if error or not path.exists():
            results[v] = {"error": error or "file not written"}
            continue

        entry = {"path": str(path)}
        entry.update(verify_tflite(path))
        try:
            coco = model.evaluate_tflite(str(path), test_loader)
            entry["coco"] = {k: float(val) for k, val in coco.items()}
            log.info("%s COCO: AP=%.4f AP50=%.4f", v, entry["coco"].get("AP", float("nan")), entry["coco"].get("AP50", float("nan")))
        except Exception as exc:  # noqa: BLE001
            log.warning("evaluate_tflite failed for %s: %s", v, exc)
        try:
            entry["pr"] = metrics.evaluate_model_file(path, test_dir, cfg.CLASS_NAMES,
                                                      negatives_dir=cfg.NEGATIVES_DIR)
        except Exception as exc:  # noqa: BLE001
            log.warning("per-class P/R failed for %s: %s", v, exc)
        results[v] = entry
    return results


def pick_deploy_variant(results: Dict[str, dict]) -> Optional[str]:
    """Prefer INT8 -> float16 -> float32, subject to size and AP-regression gates."""
    ref_ap = results.get("float32", {}).get("coco", {}).get("AP")
    for v in ("int8", "float16", "float32"):
        r = results.get(v)
        if not r or "error" in r:
            continue
        if r["size_mb"] > cfg.MODEL_SIZE_BUDGET_MB:
            log.warning("skip %s as deploy candidate: %.1f MB over budget", v, r["size_mb"])
            continue
        ap = r.get("coco", {}).get("AP")
        if v != "float32" and ref_ap is not None and ap is not None and (ref_ap - ap) > cfg.MAX_QUANT_AP_DROP:
            log.warning("skip %s: AP %.4f is %.4f below float32 (limit %.4f)", v, ap, ref_ap - ap, cfg.MAX_QUANT_AP_DROP)
            continue
        return v
    fallback = [v for v, r in results.items() if "error" not in r]
    if fallback:
        log.warning("no variant passed the gates; falling back to %s", fallback[-1])
        return fallback[-1]
    return None


def package_deploy(results: Dict[str, dict], variant: str, export_dir: Path = cfg.EXPORT_DIR) -> Path:
    """Copy the chosen model + labels + manifest into export/deploy/ (what you scp to the Pi 5)."""
    deploy = export_dir / "deploy"
    deploy.mkdir(parents=True, exist_ok=True)
    src = Path(results[variant]["path"])
    dst = deploy / cfg.DEPLOY_MODEL_NAME
    shutil.copy2(str(src), str(dst))
    (deploy / cfg.DEPLOY_LABELS_NAME).write_text("\n".join(cfg.CLASS_NAMES) + "\n")
    manifest = {
        "project": {
            "sih_problem_statement_id": cfg.PROBLEM_STATEMENT_ID,
            "team_id": cfg.TEAM_ID,
            "team_name": cfg.TEAM_NAME,
            "target_hardware": "Raspberry Pi 5 + ESP32-S3 + OpenMV + TMC2209 + Dual Flaps",
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": variant, "sha256": _sha256(dst), "size_mb": results[variant]["size_mb"],
        "io": results[variant]["io"], "class_names": cfg.CLASS_NAMES,
        "bmw_2016_bins": {
            code: {"color": prof.color_name, "category": prof.bmw_category}
            for code, prof in fcfg.BIN_PROFILES.items()
        },
        "class_to_bin": cfg.CLASS_TO_BIN,
        "fallback_highest_hazard_bin": cfg.FALLBACK_BIN,
        "recommended_thresholds": {
            "display": cfg.DISPLAY_SCORE_THRESHOLD,
            "trigger": cfg.TRIGGER_CONFIDENCE,
            "ambiguous_fail_safe": cfg.AMBIGUOUS_CONFIDENCE_FLOOR,
        },
        "coco": results[variant].get("coco"),
    }
    (deploy / cfg.DEPLOY_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    log.info("deploy package -> %s  (copy this folder to the Raspberry Pi 5)", deploy)
    return deploy


def run_export_pipeline(model, variants: Sequence[str], test_loader, test_dir: Path) -> Dict[str, dict]:
    calib = build_calibration_loader() if "int8" in variants else None
    results = export_variants(model, variants, calib, test_loader, test_dir)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORT_DIR / "export_report.json").write_text(json.dumps(results, indent=2, default=str))
    best = pick_deploy_variant(results)
    if best:
        log.info("deploy variant: %s", best)
        package_deploy(results, best)
    else:
        log.error("nothing exported successfully - see export_report.json")
    return results


# --------------------------------------------------------------------------- #
# Standalone use (EXPERIMENTAL restore path)
# --------------------------------------------------------------------------- #
def restore_model():
    """Rebuild the Model Maker detector and load the best checkpoint written by train.py."""
    import tensorflow as tf
    from tflite_model_maker import object_detector
    import train as train_mod

    train_loader = make_loader(cfg.PREPARED_SPLITS["train"])
    spec = train_mod.build_spec()
    model = object_detector.create(train_loader, model_spec=spec, batch_size=cfg.BATCH_SIZE,
                                   epochs=1, train_whole_model=cfg.TRAIN_WHOLE_MODEL, do_train=False)
    ckpt = tf.train.latest_checkpoint(str(cfg.MODEL_DIR / "best"))
    if not ckpt:
        raise FileNotFoundError("no best checkpoint under %s - run train.py first" % (cfg.MODEL_DIR / "best"))
    model.model.load_weights(ckpt).expect_partial()
    log.info("restored weights from %s", ckpt)
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-checkpoint", action="store_true", help="restore best weights (experimental)")
    ap.add_argument("--variants", nargs="+", default=list(cfg.EXPORT_VARIANTS),
                    choices=["float32", "float16", "dynamic_range", "int8"])
    ap.add_argument("--verify-only", type=Path, default=None, help="just verify an existing .tflite")
    args = ap.parse_args(argv)

    if args.verify_only:
        print(json.dumps(verify_tflite(args.verify_only), indent=2))
        return 0
    if not args.from_checkpoint:
        log.error("Nothing to export from. Train first (train.py exports automatically), "
                  "or use --from-checkpoint / --verify-only.")
        return 2
    model = restore_model()
    test_dir = cfg.PREPARED_SPLITS["test"]
    run_export_pipeline(model, args.variants, make_loader(test_dir), test_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
