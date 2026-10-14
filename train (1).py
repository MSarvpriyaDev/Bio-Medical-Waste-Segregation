#!/usr/bin/env python3
"""Train + evaluate EfficientDet-Lite0 with tflite-model-maker, then export
for the Smart Mobile Medical-Waste Collection and Segregation System
(SIH 2026 - PS ID 26115 - Team Pravah77).

    python dataset_prep.py build      # once
    python train.py                   # train -> mAP -> export float32/float16/int8 -> deploy/
    python train.py --no-export       # train + mAP + float32 reference only

Callbacks
---------
tflite-model-maker's ``create()`` assembles its own Keras callbacks internally
(periodic checkpoints, TensorBoard, COCO eval) and does not accept extra ones.
``injected_callbacks`` splices ours into the library's ``get_callbacks`` for the
duration of training:

* ModelCheckpoint(save_best_only)  -> artifacts/model_dir/best/
* EarlyStopping(restore_best_weights=True), so what gets exported is the best epoch
* LR scheduling: Model Maker's built-in cosine decay with linear warm-up, driven by
  the ``learning_rate`` / ``lr_warmup_*`` hparams below. ReduceLROnPlateau is
  intentionally NOT used: the optimiser's LR is a schedule object and Keras'
  plateau callback cannot overwrite it (it raises).

The splice touches library internals. If it does not take effect the run still
trains - the script then says loudly that early stopping / best-checkpointing
were inactive instead of failing silently.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from typing import List, Optional, Sequence

import config as cfg
from dataset_prep import make_loader

log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Spec / hyper-parameters
# --------------------------------------------------------------------------- #
def build_spec(epochs: int = cfg.EPOCHS, batch_size: int = cfg.BATCH_SIZE):
    from tflite_model_maker import object_detector  # lazy

    lr = cfg.BASE_LEARNING_RATE * batch_size / cfg.LR_REFERENCE_BATCH
    hparams = {
        "learning_rate": lr,
        "lr_warmup_init": lr * cfg.LR_WARMUP_FRACTION,
        "lr_warmup_epoch": cfg.LR_WARMUP_EPOCHS,
        "lr_decay_method": "cosine",
    }
    kwargs = dict(model_dir=str(cfg.MODEL_DIR), epochs=epochs, batch_size=batch_size,
                  tflite_max_detections=cfg.MAX_DETECTIONS, tf_random_seed=cfg.SEED)
    try:
        spec = object_detector.EfficientDetLite0Spec(hparams=hparams, **kwargs)
    except (KeyError, ValueError, TypeError) as exc:  # unknown hparam key in some versions
        log.warning("hparams %s rejected (%s) - using library default schedule", sorted(hparams), exc)
        spec = object_detector.EfficientDetLite0Spec(**kwargs)
    log.info("lr=%.4f (batch %d), warm-up %.1f epochs, cosine decay", lr, batch_size, cfg.LR_WARMUP_EPOCHS)
    return spec


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def make_callbacks() -> List:
    import tensorflow as tf

    class _Fired(tf.keras.callbacks.Callback):
        """Lets us verify that the splice actually reached Keras' fit()."""
        fired = False

        def on_train_begin(self, logs=None):
            _Fired.fired = True

    class _EarlyStopping(tf.keras.callbacks.EarlyStopping):
        """EarlyStopping that falls back to val_loss if val_det_loss is not reported."""

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None and self.monitor not in logs and "val_loss" in logs:
                log.warning("'%s' not in logs; monitoring val_loss instead", self.monitor)
                self.monitor = "val_loss"
                self.monitor_op = tf.less
            super().on_epoch_end(epoch, logs)

    best_dir = cfg.MODEL_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    return [
        _Fired(),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_dir / "ckpt"), monitor="val_det_loss", mode="min",
                                           save_best_only=True, save_weights_only=True, verbose=1),
        _EarlyStopping(monitor="val_det_loss", mode="min", patience=cfg.EARLY_STOP_PATIENCE,
                       min_delta=cfg.EARLY_STOP_MIN_DELTA, restore_best_weights=True, verbose=1),
    ]


@contextlib.contextmanager
def injected_callbacks(extra: Sequence):
    """Temporarily wrap Model Maker's EfficientDet ``train_lib.get_callbacks``."""
    patched, original, train_lib = False, None, None
    try:
        from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib  # type: ignore
        original = train_lib.get_callbacks

        def _wrapped(*args, **kwargs):
            return list(original(*args, **kwargs)) + list(extra)

        train_lib.get_callbacks = _wrapped
        patched = True
    except Exception as exc:  # noqa: BLE001
        log.warning("could not inject callbacks (%s: %s) - training with library defaults only",
                    type(exc).__name__, exc)
    try:
        yield patched
    finally:
        if patched and train_lib is not None:
            train_lib.get_callbacks = original


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _jsonable(d: dict) -> dict:
    return {k: float(v) for k, v in d.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--no-export", action="store_true", help="export only the float32 reference")
    args = ap.parse_args(argv)

    for split in ("train", "val", "test"):
        if not any(cfg.PREPARED_SPLITS[split].glob("*.xml")):
            log.error("%s is empty - run `python dataset_prep.py build` first", cfg.PREPARED_SPLITS[split])
            return 1

    import numpy as np
    import tensorflow as tf
    from tflite_model_maker import object_detector
    import export_tflite

    np.random.seed(cfg.SEED)
    tf.random.set_seed(cfg.SEED)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = make_loader(cfg.PREPARED_SPLITS["train"])
    val_data = make_loader(cfg.PREPARED_SPLITS["val"])
    test_data = make_loader(cfg.PREPARED_SPLITS["test"])
    log.info("[Team %s | PS %s] train=%d val=%d test=%d images | BMW Bins: %s",
             cfg.TEAM_NAME, cfg.PROBLEM_STATEMENT_ID,
             train_data.size, val_data.size, test_data.size, cfg.CLASS_TO_BIN)

    spec = build_spec(args.epochs, args.batch_size)
    callbacks = make_callbacks()
    with injected_callbacks(callbacks) as patched:
        model = object_detector.create(train_data, model_spec=spec, validation_data=val_data,
                                       epochs=args.epochs, batch_size=args.batch_size,
                                       train_whole_model=cfg.TRAIN_WHOLE_MODEL)
    fired = type(callbacks[0]).fired
    if not (patched and fired):
        log.warning("EARLY STOPPING / BEST-CHECKPOINT WERE INACTIVE for this run (callback injection "
                    "did not reach fit()). The last-epoch weights were exported.")

    # ---- COCO mAP (IoU 0.5 and 0.5:0.95) from Model Maker's evaluator
    coco = model.evaluate(test_data, batch_size=args.batch_size)
    (cfg.REPORT_DIR / "coco_metrics.json").write_text(json.dumps(_jsonable(coco), indent=2))
    log.info("TEST  mAP@[.5:.95]=%.4f  mAP@.5=%.4f  mAP@.75=%.4f", coco.get("AP", float("nan")),
             coco.get("AP50", float("nan")), coco.get("AP75", float("nan")))
    for k, v in coco.items():
        if k.startswith("AP_/"):
            cls_name = k[4:]
            bcode = cfg.CLASS_TO_BIN.get(cls_name, cfg.FALLBACK_BIN)
            log.info("  AP[.5:.95] %-12s (Bin %s) %.4f", cls_name, bcode, v)

    # ---- export (also yields per-class precision/recall on the exported models)
    variants = ("float32",) if args.no_export else cfg.EXPORT_VARIANTS
    results = export_tflite.run_export_pipeline(model, variants, test_data, cfg.PREPARED_SPLITS["test"])

    import metrics
    for v in variants:
        pr = results.get(v, {}).get("pr")
        if pr:
            print("\n===== %s =====" % v)
            print(metrics.format_report(pr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
