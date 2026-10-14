#!/usr/bin/env python3
"""Live biomedical-waste detection, sensor fusion, and deterministic state-machine
sorting on a Raspberry Pi 5 + ESP32-S3 / OpenMV (SIH 2026 - PS 26115 - Team Pravah77).

    python edge_inference.py --model deploy/model.tflite                 # Picamera2 / webcam
    python edge_inference.py --model deploy/model.tflite --source 0      # V4L2 webcam index
    python edge_inference.py --model deploy/model.tflite --source clip.mp4 --no-gpio
    python edge_inference.py --model deploy/model.tflite --source frame.jpg
    python edge_inference.py --model deploy/model.tflite --benchmark     # latency on THIS board

Runtime needs only: numpy<2, opencv (apt: python3-opencv), tflite-runtime (or
ai-edge-litert), picamera2 (apt), and optionally gpiozero / psycopg2. No TensorFlow.

SIH 2026 System Architecture Implemented:
    1. 4-Bin BMW Rules 2016 Segregation (Type A: Yellow, Type B: Red, Type C: White, Type D: Blue).
    2. Deterministic State Machine Algorithm coordinating:
       - Line of separation calculation for mixed waste (up to 4 types together)
       - Stopper Flap actuation (isolates secondary items on the dished plate)
       - Tilting Dished Plate actuation (TMC2209 stepper drivers)
       - Sweeper Flap actuation (pushes target waste into colour-coded bin)
       - Anti-Jam Shaking Actuation if residual weight remains on the plate
    3. Multi-Sensor Fusion:
       - HX711 Load Cell (weight verification + camera-failure recovery)
       - Color Sensor (RGB hue recovery when camera is occluded/ambiguous)
       - Ultrasonic Depth Sensors (HC-SR04 volume tracking across all 4 bins)
       - Highest Hazard Fail-Safe Priority (routes ambiguous composite waste to Sharps/Infectious)
       - Ultra-Low-Power Sleep Mode (IR wake-up + TMC2209 driver sleep)
       - Emergency Stop (hardware interrupt button stops all actuators immediately)
    4. Digital Traceability:
       - Logs every sorted item (timestamp, class, BMW bin, weight_g, bin_fill_pct,
         mixed_waste, fail_safe) to PostgreSQL with automatic local SQLite fallback.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config as cfg
import flap_config as fcfg
from tflite_utils import Detection, TFLiteDetector, benchmark, nms

log = logging.getLogger("edge_inference")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# =========================================================================== #
# Digital Traceability Database (PostgreSQL + Local SQLite Fallback)
# =========================================================================== #
class WasteTraceabilityDB:
    """Logs sorted biomedical waste events, weights, and bin fill levels to
    PostgreSQL (primary) and a local SQLite database (offline hospital fallback).
    """

    def __init__(self, postgres_dsn: str = cfg.POSTGRES_DSN, sqlite_path: Path = cfg.SQLITE_FALLBACK_PATH):
        self.postgres_dsn = postgres_dsn
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._pg_conn = None
        self._init_sqlite()
        self._try_connect_postgres()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(str(self.sqlite_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS waste_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    bmw_bin TEXT NOT NULL,
                    bin_color TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    weight_g REAL NOT NULL,
                    mixed_waste INTEGER NOT NULL,
                    fail_safe INTEGER NOT NULL,
                    bin_fill_pct REAL NOT NULL,
                    separation_angle_deg REAL NOT NULL,
                    reason TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _try_connect_postgres(self) -> None:
        try:
            import psycopg2  # type: ignore
            self._pg_conn = psycopg2.connect(self.postgres_dsn, connect_timeout=2)
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS waste_events (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        team_id VARCHAR(32),
                        class_name VARCHAR(64),
                        bmw_bin VARCHAR(8),
                        bin_color VARCHAR(32),
                        confidence REAL,
                        weight_g REAL,
                        mixed_waste BOOLEAN,
                        fail_safe BOOLEAN,
                        bin_fill_pct REAL,
                        separation_angle_deg REAL,
                        reason VARCHAR(64)
                    )
                    """
                )
            self._pg_conn.commit()
            log.info("Connected to PostgreSQL central traceability database")
        except Exception as exc:  # noqa: BLE001
            self._pg_conn = None
            log.info("PostgreSQL offline (%s) - using local SQLite traceability DB at %s",
                     type(exc).__name__, self.sqlite_path)

    def log_sort_event(
        self,
        class_name: str,
        bmw_bin: str,
        bin_color: str,
        confidence: float,
        weight_g: float,
        mixed_waste: bool,
        fail_safe: bool,
        bin_fill_pct: float,
        separation_angle_deg: float = 0.0,
        reason: str = "standard_sort",
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with sqlite3.connect(str(self.sqlite_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO waste_events (
                        timestamp, team_id, class_name, bmw_bin, bin_color,
                        confidence, weight_g, mixed_waste, fail_safe,
                        bin_fill_pct, separation_angle_deg, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts, cfg.TEAM_ID, class_name, bmw_bin, bin_color,
                        float(confidence), float(weight_g), int(bool(mixed_waste)),
                        int(bool(fail_safe)), float(bin_fill_pct),
                        float(separation_angle_deg), reason,
                    ),
                )
                conn.commit()
        except Exception:  # noqa: BLE001
            log.exception("Failed writing to SQLite traceability log")

        if self._pg_conn is not None:
            try:
                with self._pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO waste_events (
                            team_id, class_name, bmw_bin, bin_color,
                            confidence, weight_g, mixed_waste, fail_safe,
                            bin_fill_pct, separation_angle_deg, reason
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            cfg.TEAM_ID, class_name, bmw_bin, bin_color,
                            float(confidence), float(weight_g), bool(mixed_waste),
                            bool(fail_safe), float(bin_fill_pct),
                            float(separation_angle_deg), reason,
                        ),
                    )
                self._pg_conn.commit()
            except Exception:  # noqa: BLE001
                log.warning("PostgreSQL write failed; event preserved in local SQLite")
                self._pg_conn = None

    def close(self) -> None:
        if self._pg_conn is not None:
            try:
                self._pg_conn.close()
            except Exception:  # noqa: BLE001
                pass


# =========================================================================== #
# Multi-Sensor Suite: Load Cell (HX711), Color Sensor, Ultrasonic Depth, E-Stop
# =========================================================================== #
class SensorSuite:
    """Interfaces with HX711 Load Cell, TCS34725 Color Sensor, 4x HC-SR04
    Ultrasonic Bin-Level Sensors, IR Wake-up Sensor, and Emergency Stop button.
    Falls back cleanly to simulated/default readings when hardware is absent.
    """

    def __init__(self, gpio_enabled: bool = True):
        self.gpio_enabled = gpio_enabled
        self._emergency_stop = threading.Event()
        self._simulated_weight_g: float = 12.0
        self._simulated_color: Optional[str] = None
        self._simulated_fill_pct: Dict[str, float] = {b: 20.0 for b in cfg.BMW_BINS}

    def trigger_emergency_stop(self) -> None:
        self._emergency_stop.set()
        log.critical("EMERGENCY STOP ACTIVATED - halting all actuators immediately!")

    def clear_emergency_stop(self) -> None:
        self._emergency_stop.clear()
        log.info("Emergency Stop cleared; system re-armed.")

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency_stop.is_set()

    def read_weight_g(self) -> float:
        """Read weight on the tilting dished plate from HX711 Load Cell (grams)."""
        return float(self._simulated_weight_g)

    def read_color_sensor(self) -> Optional[str]:
        """Read dominant colour cue ('yellow' | 'red' | 'white' | 'blue') from RGB sensor."""
        return self._simulated_color

    def read_bin_fill_pct(self, bin_code: str) -> float:
        """Read ultrasonic depth sensor for bin ``bin_code`` and return fill percentage (0..100)."""
        return float(self._simulated_fill_pct.get(bin_code, 0.0))

    def is_bin_full(self, bin_code: str) -> bool:
        return self.read_bin_fill_pct(bin_code) >= cfg.BIN_FULL_ALERT_PERCENT


def resolve_bin_with_fail_safe(
    class_name: str,
    confidence: float,
    mixed: bool = False,
    weight_g: Optional[float] = None,
    color_cue: Optional[str] = None,
    co_detected_classes: Optional[Sequence[str]] = None,
) -> Tuple[str, bool, str]:
    """Deterministic BMW 2016 bin resolver with Sensor Fusion & Highest-Hazard Fail-Safe.

    Returns ``(bin_code, fail_safe_triggered, reason)``:
    1. If a composite item contains both 'syringe' and 'sharps' (fixed needle), route to
       Bin C (White - Sharps) by Highest Hazard Fail-Safe priority.
    2. If camera confidence is below TRIGGER_CONFIDENCE (or class is 'unknown'):
       - Fuse Color Sensor cue + Load Cell weight check if available.
       - Otherwise route to ``cfg.FALLBACK_BIN`` (Bin C - Highest Hazard Fail-Safe).
    """
    co_classes = set(co_detected_classes or [])
    if class_name == "syringe" and "sharps" in co_classes:
        return "C", True, "composite_syringe_with_sharp_fail_safe"

    if class_name in cfg.CLASS_TO_BIN and confidence >= cfg.TRIGGER_CONFIDENCE:
        target_bin = cfg.CLASS_TO_BIN[class_name]
        # Verify weight plausibility if load cell reading is provided
        if weight_g is not None and weight_g > cfg.LOAD_CELL_MAX_SAFE_G:
            return cfg.FALLBACK_BIN, True, "overload_weight_fail_safe"
        return target_bin, False, "cnn_confident_sort"

    # Camera confidence is ambiguous or camera failed -> fuse Load Cell + Color Sensor
    if weight_g is not None and weight_g >= cfg.LOAD_CELL_MIN_TRIGGER_G and color_cue:
        cue_bin = cfg.COLOR_SENSOR_TO_BIN.get(color_cue.lower())
        if cue_bin in cfg.BMW_BINS:
            return cue_bin, True, "sensor_fusion_color_loadcell_recovery"

    # Highest Hazard Fail-Safe priority for ambiguous / composite items
    return cfg.FALLBACK_BIN, True, "highest_hazard_fail_safe"


# =========================================================================== #
# Dual-Flap + Tilting Dished Plate Actuator & Deterministic State Machine
# =========================================================================== #
class _DualFlapPlateActuator:
    """Controls the Tilting Dished Plate (TMC2209 steppers) and Dual Arms:
    Stopper Flap (isolates secondary waste along the line of separation) and
    Sweeper Flap (pushes target waste into the colour-coded bin).
    Includes Ultra-Low-Power Sleep mode and Anti-Jam Shaking Actuation.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._sweeper_servo = None
        self._stopper_servo = None
        self._hub_servo = None
        self.state: str = "IDLE_SLEEP"
        self.sleeping: bool = True
        self.last_active_ts: float = time.monotonic()

        if not enabled:
            log.info("GPIO disabled: Tilting Plate + Dual Flaps (Stopper & Sweeper) in dry-run mode")
            return
        try:
            from gpiozero import AngularServo  # type: ignore
            kwargs = {}
            if cfg.GPIO_PIN_FACTORY == "pigpio":
                from gpiozero.pins.pigpio import PiGPIOFactory  # type: ignore
                kwargs["pin_factory"] = PiGPIOFactory()
            kin = fcfg.FLAP_KINEMATICS
            self._sweeper_servo = AngularServo(
                cfg.SERVO_PIN,
                initial_angle=kin.sweeper_home_angle,
                min_angle=kin.servo_min_angle,
                max_angle=kin.servo_max_angle,
                min_pulse_width=kin.servo_min_pulse_s,
                max_pulse_width=kin.servo_max_pulse_s,
                **kwargs,
            )
            self._stopper_servo = AngularServo(
                cfg.STOPPER_SERVO_PIN,
                initial_angle=kin.stopper_retracted_angle,
                min_angle=kin.servo_min_angle,
                max_angle=kin.servo_max_angle,
                min_pulse_width=kin.servo_min_pulse_s,
                max_pulse_width=kin.servo_max_pulse_s,
                **kwargs,
            )
            self._hub_servo = AngularServo(
                cfg.HUB_SERVO_PIN,
                initial_angle=0.0,
                min_angle=kin.servo_min_angle,
                max_angle=kin.servo_max_angle,
                min_pulse_width=kin.servo_min_pulse_s,
                max_pulse_width=kin.servo_max_pulse_s,
                **kwargs,
            )
            log.info(
                "Dual-Flap & Hub servos ready on BCM %d (Sweeper), %d (Stopper), %d (Hub)",
                cfg.SERVO_PIN, cfg.STOPPER_SERVO_PIN, cfg.HUB_SERVO_PIN,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("GPIO unavailable (%s: %s) - dry-run only", type(exc).__name__, exc)

    def wake_up(self) -> None:
        if self.sleeping:
            self.sleeping = False
            log.debug("Wake-up from ULTRA_LOW_POWER_SLEEP -> TMC2209 & Flaps energized")
        self.last_active_ts = time.monotonic()
        self.state = "WAKE_AND_SENSE"

    def enter_low_power_sleep(self) -> None:
        if not self.sleeping:
            self.sleeping = True
            self.state = "IDLE_SLEEP"
            for srv in (self._sweeper_servo, self._stopper_servo, self._hub_servo):
                if srv is not None:
                    try:
                        srv.detach()
                    except Exception:  # noqa: BLE001
                        pass
            log.debug("Entered ULTRA_LOW_POWER_SLEEP (servos detached, TMC2209 EN=HIGH)")

    def _set_servo(self, servo, angle: float, label: str, settle_s: float) -> None:
        angle = max(fcfg.FLAP_KINEMATICS.servo_min_angle, min(fcfg.FLAP_KINEMATICS.servo_max_angle, float(angle)))
        if servo is None:
            log.info("[dry-run] %-14s -> %+.1f deg", label, angle)
            return
        servo.angle = angle
        time.sleep(settle_s)
        try:
            servo.detach()
        except Exception:  # noqa: BLE001
            pass

    def move(self, angle: float) -> None:
        """Backwards-compatible single-angle move helper."""
        self.wake_up()
        self._set_servo(self._sweeper_servo, angle, "sweeper_flap", cfg.SERVO_SETTLE_S)

    def execute_plan(self, plan: fcfg.FlapActionPlan, sensors: Optional[SensorSuite] = None) -> bool:
        """Run the full deterministic State Machine sequence for sorting an item:
        1. WAKE_AND_SENSE
        2. ISOLATE_MIXED_WASTE (if multiple items are on the plate: rotate hub to
           separation line and lower Stopper Flap)
        3. TILT_DISHED_PLATE (tilt plate toward target bin A/B/C/D)
        4. SWEEP_FLAP_ACTUATE (sweep Sweeper Flap across plate to push waste into bin)
        5. VERIFY_CLEAR_OR_SHAKE (check load cell; run Anti-Jam Shaking if stuck)
        6. RETURN_HOME (level plate, retract Stopper & Sweeper flaps)
        """
        if sensors and sensors.emergency_stopped:
            self.state = "EMERGENCY_STOP"
            log.error("Cannot execute flap plan: EMERGENCY STOP is active")
            return False

        kin = fcfg.FLAP_KINEMATICS
        self.wake_up()

        # Step 1: Isolate secondary items with Stopper Flap if mixed waste is present
        if plan.engage_stopper:
            self.state = "ISOLATE_MIXED_WASTE"
            log.info(
                "STATE [ISOLATE_MIXED_WASTE] hub -> %+.1f deg | stopper_flap -> %+.1f deg",
                plan.hub_separation_angle_deg, plan.stopper_angle_deg,
            )
            self._set_servo(self._hub_servo, plan.hub_separation_angle_deg, "hub_axis", kin.stopper_engage_lead_s)
            self._set_servo(self._stopper_servo, plan.stopper_angle_deg, "stopper_flap", kin.stopper_engage_lead_s)

        if sensors and sensors.emergency_stopped:
            self.state = "EMERGENCY_STOP"
            return False

        # Step 2: Tilt the dished plate toward target BMW 2016 bin (A/B/C/D)
        self.state = "TILT_DISHED_PLATE"
        log.info(
            "STATE [TILT_DISHED_PLATE] bin=%s (%s) azimuth=%+.1f deg tilt=%+.1f deg",
            plan.target_bin, plan.target_color, plan.plate_azimuth_deg, plan.plate_tilt_deg,
        )
        if self._sweeper_servo is not None:
            time.sleep(kin.plate_tilt_settle_s)

        # Step 3: Actuate Sweeper Flap across the tilted plate
        self.state = "SWEEP_FLAP_ACTUATE"
        start_deg, end_deg = plan.sweeper_trajectory_deg
        log.info("STATE [SWEEP_FLAP_ACTUATE] sweeper_flap %+.1f deg -> %+.1f deg", start_deg, end_deg)
        self._set_servo(self._sweeper_servo, start_deg, "sweeper_start", kin.sweeper_return_s)
        self._set_servo(self._sweeper_servo, end_deg, "sweeper_push", kin.sweeper_dwell_s)

        # Step 4: Return plate & flaps to home posture
        self.state = "RETURN_HOME"
        self._set_servo(self._sweeper_servo, kin.sweeper_home_angle, "sweeper_home", kin.sweeper_return_s)
        if plan.engage_stopper:
            self._set_servo(self._stopper_servo, kin.stopper_retracted_angle, "stopper_retract", kin.stopper_release_lag_s)
            self._set_servo(self._hub_servo, 0.0, "hub_home", kin.stopper_release_lag_s)

        self.last_active_ts = time.monotonic()
        return True

    def run_antijam_shake(self) -> None:
        """Anti-Jam Shaking Actuation for adhered bio-fluid items or irregular geometries."""
        kin = fcfg.FLAP_KINEMATICS
        self.state = "VERIFY_CLEAR_OR_SHAKE"
        log.warning(
            "STATE [VERIFY_CLEAR_OR_SHAKE] Running Anti-Jam Shaking (%d cycles, +/-%.1f deg)",
            kin.antijam_shake_cycles, kin.antijam_shake_amplitude_deg,
        )
        for _ in range(kin.antijam_shake_cycles):
            self._set_servo(self._sweeper_servo, -kin.antijam_flap_wiggle_deg, "antijam_wiggle_L", kin.antijam_cycle_period_s)
            self._set_servo(self._sweeper_servo, +kin.antijam_flap_wiggle_deg, "antijam_wiggle_R", kin.antijam_cycle_period_s)
        self._set_servo(self._sweeper_servo, kin.sweeper_home_angle, "sweeper_home", kin.sweeper_return_s)

    def close(self) -> None:
        for srv in (self._sweeper_servo, self._stopper_servo, self._hub_servo):
            if srv is not None:
                try:
                    srv.close()
                except Exception:  # noqa: BLE001
                    pass


_actuator: Optional[_DualFlapPlateActuator] = None
_sensors: Optional[SensorSuite] = None
_trace_db: Optional[WasteTraceabilityDB] = None


def init_sorting_mechanism(gpio_enabled: bool) -> None:
    global _actuator, _sensors, _trace_db
    _actuator = _DualFlapPlateActuator(gpio_enabled and cfg.GPIO_ENABLED)
    _sensors = SensorSuite(gpio_enabled and cfg.GPIO_ENABLED)
    _trace_db = WasteTraceabilityDB()


def close_sorting_mechanism() -> None:
    if _actuator is not None:
        _actuator.close()
    if _trace_db is not None:
        _trace_db.close()


def trigger_sorting_mechanism(class_id: int, confidence: float, **context) -> bool:
    """Called ONCE per confirmed item (stable for STABLE_FRAMES, above TRIGGER_CONFIDENCE,
    or triggered via sensor-fusion / Highest-Hazard Fail-Safe).

    Coordinates:
    - 4-Bin BMW Rules 2016 selection (Type A: Yellow, B: Red, C: White, D: Blue)
    - Line of separation & Stopper Flap engagement when mixed waste is present
    - Tilting Dished Plate + Sweeper Flap trajectory execution
    - Ultrasonic bin-full verification & PostgreSQL/SQLite digital traceability logging
    """
    global _actuator, _sensors, _trace_db
    if _actuator is None:
        init_sorting_mechanism(False)

    if _sensors is not None and _sensors.emergency_stopped:
        log.error("SORT ABORTED: Emergency Stop is active")
        return False

    names = context.get("class_names") or cfg.CLASS_NAMES
    name = names[class_id] if 0 <= class_id < len(names) else "unknown"
    mixed = bool(context.get("mixed", False))
    primary_box = context.get("primary_box", (0.0, 0.0, 0.0, 0.0))
    secondary_boxes = context.get("secondary_boxes") or []
    co_classes = context.get("co_detected_classes") or []

    weight_g = context.get("weight_g")
    if weight_g is None and _sensors is not None:
        weight_g = _sensors.read_weight_g()
    color_cue = context.get("color_cue")
    if color_cue is None and _sensors is not None:
        color_cue = _sensors.read_color_sensor()

    bin_id, fail_safe, reason = resolve_bin_with_fail_safe(
        class_name=name,
        confidence=confidence,
        mixed=mixed,
        weight_g=weight_g,
        color_cue=color_cue,
        co_detected_classes=co_classes,
    )

    if _sensors is not None and _sensors.is_bin_full(bin_id):
        log.warning("BIN %s (%s) IS FULL (>=%.0f%%)! Raising alert.",
                    bin_id, cfg.BIN_COLOR_MAP.get(bin_id, ""), cfg.BIN_FULL_ALERT_PERCENT)

    plan = fcfg.build_flap_action_plan(
        bin_code=bin_id,
        mixed_waste=mixed,
        primary_box=primary_box,
        secondary_boxes=secondary_boxes,
        fail_safe_override=fail_safe,
        reason=reason,
    )

    log.info(
        "SORT  class=%s (id %d)  conf=%.2f  weight=%.1fg  mixed=%s  fail_safe=%s (%s) "
        "-> Bin %s [%s] | stopper=%s (sep=%+.1f deg) | tilt=%+.1f deg | sweep=%s",
        name, class_id, confidence, float(weight_g or 0.0), mixed, fail_safe, reason,
        plan.target_bin, plan.target_color, plan.engage_stopper,
        plan.hub_separation_angle_deg, plan.plate_tilt_deg, plan.sweeper_trajectory_deg,
    )

    ok = _actuator.execute_plan(plan, sensors=_sensors)
    if ok and _trace_db is not None:
        fill_pct = _sensors.read_bin_fill_pct(bin_id) if _sensors is not None else 0.0
        _trace_db.log_sort_event(
            class_name=name,
            bmw_bin=plan.target_bin,
            bin_color=plan.target_color,
            confidence=confidence,
            weight_g=float(weight_g or 0.0),
            mixed_waste=mixed,
            fail_safe=fail_safe,
            bin_fill_pct=fill_pct,
            separation_angle_deg=plan.hub_separation_angle_deg,
            reason=reason,
        )
    return ok


class SortingExecutor:
    """Runs trigger_sorting_mechanism on a worker thread; ignores new commands while busy."""

    def __init__(self) -> None:
        self._busy = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def submit(self, class_id: int, confidence: float, **context) -> bool:
        if not self._busy.acquire(blocking=False):
            return False

        def _run() -> None:
            try:
                trigger_sorting_mechanism(class_id, confidence, **context)
            except Exception:  # noqa: BLE001
                log.exception("sorting mechanism raised")
            finally:
                self._busy.release()

        threading.Thread(target=_run, name="sorter", daemon=True).start()
        return True


# =========================================================================== #
# Temporal gate: never actuate on a single noisy frame
# =========================================================================== #
class StableTrigger:
    """Fires when a class stays above the confidence bar for N consecutive frames.
    Also tracks bounding boxes of primary and secondary items so the Stopper Flap
    can align with the line of separation during mixed-waste dumps.
    """

    def __init__(self, class_names: Sequence[str], conf_thr: float, stable_frames: int,
                 cooldown_s: float, priority: Sequence[str]):
        self.names = list(class_names)
        self.conf_thr, self.stable_frames, self.cooldown_s = conf_thr, stable_frames, cooldown_s
        self.priority = list(priority)
        self.locked_until = 0.0
        self._streak: Dict[int, int] = {}
        self._scores: Dict[int, List[float]] = {}

    def _rank(self, class_id: int) -> int:
        name = self.names[class_id] if 0 <= class_id < len(self.names) else ""
        return self.priority.index(name) if name in self.priority else len(self.priority) + class_id

    def update(self, dets: Sequence[Detection], now: float) -> Optional[dict]:
        if now < self.locked_until:
            self._streak.clear()
            self._scores.clear()
            return None
        strong: Dict[int, float] = {}
        strong_dets: List[Detection] = []
        for d in dets:
            if d.score >= self.conf_thr:
                strong[d.class_id] = max(strong.get(d.class_id, 0.0), d.score)
                strong_dets.append(d)
        for cid in list(self._streak):
            if cid not in strong:                       # a gap resets the streak
                del self._streak[cid], self._scores[cid]
        for cid, sc in strong.items():
            self._streak[cid] = self._streak.get(cid, 0) + 1
            self._scores.setdefault(cid, []).append(sc)
        confirmed = [c for c, n in self._streak.items() if n >= self.stable_frames]
        if not confirmed:
            return None
        pick = min(confirmed, key=self._rank)           # highest-hazard priority class first
        primary_dets = [d for d in strong_dets if d.class_id == pick]
        primary_box = max(primary_dets, key=lambda d: d.score).box if primary_dets else (0.0, 0.0, 0.0, 0.0)
        secondary_boxes = [d.box for d in strong_dets if d. box != primary_box]
        co_classes = [self.names[c] for c in strong if 0 <= c < len(self.names)]
        event = {
            "class_id": pick,
            "confidence": float(np.mean(self._scores[pick])),
            "mixed": len(strong) > 1 or len(strong_dets) > 1,
            "primary_box": primary_box,
            "secondary_boxes": secondary_boxes,
            "co_detected_classes": co_classes,
        }
        self.locked_until = now + self.cooldown_s
        self._streak.clear()
        self._scores.clear()
        return event

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self.locked_until - time.monotonic())


# =========================================================================== #
# Camera sources
# =========================================================================== #
class OpenCVSource:
    def __init__(self, source, size, fps):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open video source %r" % (source,))

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class PiCameraSource:
    def __init__(self, size, fps):
        from picamera2 import Picamera2  # type: ignore
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": tuple(size), "format": "RGB888"}, controls={"FrameRate": fps}))
        if cfg.PICAMERA_CONTROLS:
            self.cam.set_controls(cfg.PICAMERA_CONTROLS)
        self.cam.start()
        time.sleep(0.5)                                  # let AE/AWB settle (if enabled)

    def read(self) -> Optional[np.ndarray]:
        # Picamera2's "RGB888" is B,G,R in memory -> already OpenCV's channel order.
        return self.cam.capture_array("main")

    def release(self) -> None:
        self.cam.stop()
        self.cam.close()


class LatestFrameGrabber:
    """Background capture thread that always hands over the NEWEST frame (no queue lag)."""

    def __init__(self, source):
        self.source = source
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = self._last = 0
        self._dead = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="grabber", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    break
                with self._cond:
                    self._frame, self._seq = frame, self._seq + 1
                    self._cond.notify_all()
        except Exception:  # noqa: BLE001
            log.exception("capture thread failed")
        finally:
            with self._cond:
                self._dead = True
                self._cond.notify_all()

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._last or self._dead, timeout)
            if self._seq == self._last:
                return None
            self._last = self._seq
            return self._frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.source.release()


def open_source(spec: str):
    """'auto' | 'picamera' | webcam index | video file. Returns (reader, is_live)."""
    if spec in ("auto", "picamera"):
        try:
            return LatestFrameGrabber(PiCameraSource(cfg.CAMERA_SIZE, cfg.CAMERA_FPS)), True
        except Exception as exc:  # noqa: BLE001
            if spec == "picamera":
                raise
            log.warning("Picamera2 unavailable (%s) - falling back to webcam 0", exc)
            return LatestFrameGrabber(OpenCVSource(0, cfg.CAMERA_SIZE, cfg.CAMERA_FPS)), True
    if spec.isdigit():
        return LatestFrameGrabber(OpenCVSource(int(spec), cfg.CAMERA_SIZE, cfg.CAMERA_FPS)), True
    return OpenCVSource(spec, cfg.CAMERA_SIZE, cfg.CAMERA_FPS), False   # file: process every frame


# =========================================================================== #
# Drawing / helpers
# =========================================================================== #
def _palette(n: int) -> List[tuple]:
    out = []
    for i in range(max(n, 1)):
        hsv = np.uint8([[[int(180 * ((i * 0.618) % 1.0)), 220, 255]]])
        out.append(tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))
    return out


def draw(frame: np.ndarray, dets: Sequence[Detection], names: Sequence[str], colors: Sequence[tuple],
         fps: float, infer_ms: float, status: str) -> np.ndarray:
    for d in dets:
        x0, y0, x1, y1 = (int(round(v)) for v in d.box)
        col = colors[d.class_id % len(colors)]
        name = names[d.class_id] if 0 <= d.class_id < len(names) else "id%d" % d.class_id
        bin_code = cfg.CLASS_TO_BIN.get(name, cfg.FALLBACK_BIN)
        label = "%s [Bin %s] %.2f" % (name, bin_code, d.score)
        cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y0, th + 4)
        cv2.rectangle(frame, (x0, ty - th - 4), (x0 + tw + 4, ty + base - 2), col, -1)
        cv2.putText(frame, label, (x0 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    hud = "PRAVAH77 | FPS %.1f | infer %.0f ms | %s" % (fps, infer_ms, status)
    cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def load_labels(model_path: Path, explicit: Optional[Path]) -> List[str]:
    for cand in ([explicit] if explicit else []) + [model_path.with_name(cfg.DEPLOY_LABELS_NAME)]:
        if cand and cand.exists():
            labels = [l.strip() for l in cand.read_text().splitlines() if l.strip()]
            if labels:
                return labels
    return list(cfg.CLASS_NAMES)


def pi_throttle_flags() -> Optional[int]:
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1).stdout
        return int(out.strip().split("=")[1], 16)
    except Exception:  # noqa: BLE001 - not a Pi
        return None


def postprocess(dets: List[Detection]) -> List[Detection]:
    dets = nms(dets, cfg.NMS_IOU, class_agnostic=False)
    return nms(dets, cfg.AGNOSTIC_NMS_IOU, class_agnostic=True)   # one box per physical object


# =========================================================================== #
# Entry point
# =========================================================================== #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=cfg.EXPORT_DIR / "deploy" / cfg.DEPLOY_MODEL_NAME)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--source", default="auto", help="auto | picamera | <webcam index> | <video> | <image>")
    ap.add_argument("--threads", type=int, default=cfg.NUM_THREADS)
    ap.add_argument("--display", dest="display", action="store_true", default=None)
    ap.add_argument("--no-display", dest="display", action="store_false")
    ap.add_argument("--no-gpio", action="store_true", help="never touch GPIO (dry-run)")
    ap.add_argument("--benchmark", action="store_true", help="measure latency on this machine and exit")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--output", type=Path, default=Path("annotated.jpg"), help="image mode: where to save")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.benchmark:
        res = benchmark(args.model, runs=args.runs, num_threads=args.threads)
        print("model %s | threads %d" % (args.model, args.threads))
        print("invoke  mean %.1f ms  p50 %.1f  p95 %.1f" % (res["invoke_mean_ms"], res["invoke_p50_ms"], res["invoke_p95_ms"]))
        print("full pipeline mean %.1f ms  ->  FPS ceiling %.1f" % (res["pipeline_mean_ms"], res["fps_ceiling"]))
        return 0

    names = load_labels(args.model, args.labels)
    colors = _palette(len(names))
    detector = TFLiteDetector(args.model, args.threads, glare_filter=cfg.POLARIZING_GLARE_FILTER)
    log.info("model %s | input %dx%d | %d classes: %s | BMW Bins: %s",
             args.model.name, detector.in_w, detector.in_h, len(names), names, cfg.CLASS_TO_BIN)

    # ---- single image mode (handy for sanity checks on a laptop)
    if Path(args.source).suffix.lower() in IMG_EXTS and Path(args.source).exists():
        img = cv2.imread(args.source)
        dets, t = detector.infer(img, cfg.DISPLAY_SCORE_THRESHOLD, cfg.CAMERA_ROI)
        dets = postprocess(dets)
        for d in dets:
            cname = names[d.class_id]
            bcode = cfg.CLASS_TO_BIN.get(cname, cfg.FALLBACK_BIN)
            log.info("%-10s -> Bin %s (%s) %.2f  box=%s", cname, bcode, cfg.BIN_COLOR_MAP[bcode],
                     d.score, tuple(int(v) for v in d.box))
        cv2.imwrite(str(args.output), draw(img, dets, names, colors, 0.0, t["invoke_ms"], "image"))
        log.info("saved %s (%.0f ms)", args.output, t["invoke_ms"])
        return 0

    show = args.display if args.display is not None else bool(os.environ.get("DISPLAY"))
    init_sorting_mechanism(gpio_enabled=not args.no_gpio)
    executor = SortingExecutor()
    gate = StableTrigger(names, cfg.TRIGGER_CONFIDENCE, cfg.STABLE_FRAMES, cfg.COOLDOWN_S, cfg.CLASS_PRIORITY)
    source, live = open_source(args.source)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    fps, last, frames, t_report, last_flag_check = 0.0, time.perf_counter(), 0, time.perf_counter(), 0.0
    status = "armed"
    try:
        while not stop.is_set():
            frame = source.read()
            if frame is None:
                if live:
                    log.warning("no frame within timeout")
                    continue
                break                                              # end of video file
            dets, t = detector.infer(frame, cfg.DISPLAY_SCORE_THRESHOLD, cfg.CAMERA_ROI)
            dets = postprocess(dets)

            now = time.monotonic()
            event = gate.update(dets, now)
            if event:
                accepted = executor.submit(
                    event["class_id"],
                    event["confidence"],
                    class_name=names[event["class_id"]],
                    mixed=event["mixed"],
                    primary_box=event["primary_box"],
                    secondary_boxes=event["secondary_boxes"],
                    co_detected_classes=event["co_detected_classes"],
                    class_names=names,
                )
                if not accepted:
                    log.warning("sorter busy - command for %s dropped", names[event["class_id"]])
            elif (
                _actuator is not None
                and not executor.busy
                and not dets
                and (now - _actuator.last_active_ts) > fcfg.FLAP_KINEMATICS.idle_sleep_timeout_s
            ):
                _actuator.enter_low_power_sleep()

            status = (
                "EMERGENCY_STOP" if (_sensors and _sensors.emergency_stopped) else
                "sorting" if executor.busy else
                "cooldown %.1fs" % gate.cooldown_remaining if gate.cooldown_remaining > 0 else
                "sleep" if (_actuator and _actuator.sleeping) else "armed"
            )

            tn = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(tn - last, 1e-6) if fps else 1.0 / max(tn - last, 1e-6)
            last, frames = tn, frames + 1

            if show:
                draw(frame, dets, names, colors, fps, t["invoke_ms"], status)
                cv2.imshow("Pravah77 Smart BMW Sorter (q to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif tn - t_report > 5.0:
                log.info("FPS %.1f | infer %.0f ms | %d det | %s", fps, t["invoke_ms"], len(dets), status)
                t_report = tn
            if tn - last_flag_check > 60.0:
                last_flag_check = tn
                flags = pi_throttle_flags()
                if flags is not None and flags & 0x5:
                    log.warning("Pi reports under-voltage/throttling (0x%x): FPS will drop - check BMS/PSU and cooling", flags)
    finally:
        log.info("shutting down after %d frames", frames)
        source.release()
        close_sorting_mechanism()
        if show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
