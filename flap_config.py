"""Configuration and kinematics for the Tilting Dished Plate and Dual Flaps
(Stopper Flap & Sweeper Flap) for Smart Biomedical Waste Segregation (Team Pravah77).

Hardware Architecture (SIH2026 - PS 26115):
- Brain: Raspberry Pi 5 (Vision & State Machine) + ESP32-S3 (Actuator & Sensor Hub)
- Stepper Drivers: TMC2209 (2-axis dish tilt / hub rotation, plus anti-jam shaking)
- Dual Arms / Flaps:
    1. Stopper Flap (Servo): Drops down along the calculated line of separation to
       isolate and hold secondary waste items during mixed-waste dumps.
    2. Sweeper Flap (Servo): Sweeps across the tilted dished plate to push the target
       waste item into the designated BMW Rules 2016 colour-coded bin.
- Sensors: HX711 Load Cell (weight validation), TCS34725 Color Sensor (backup optical cue),
           HC-SR04 Ultrasonic Depth Sensors (bin-full tracking across 4 bins),
           IR Break-beam / Proximity (ultra-low-power wake-up), Emergency Stop (NC hardware interrupt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# BMW Rules 2016: 4 Colour-Coded Bins (Type A / B / C / D)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BinProfile:
    bin_code: str                # "A", "B", "C", "D"
    color_name: str              # Yellow, Red, White, Blue
    bmw_category: str            # Official BMW 2016 waste category description
    plate_azimuth_deg: float     # Rotational/tilt direction of the dished plate (deg)
    plate_tilt_deg: float        # Downward tilt angle of the dished plate toward bin (deg)
    sweeper_start_deg: float     # Sweeper flap home/entry angle before sweeping (deg)
    sweeper_end_deg: float       # Sweeper flap terminal push angle into the chute (deg)
    stopper_hold_deg: float      # Stopper flap barrier angle when isolating mixed waste (deg)
    ultrasonic_trig_pin: int     # BCM pin for bin volume tracking trigger
    ultrasonic_echo_pin: int     # BCM pin for bin volume tracking echo


BIN_PROFILES: Dict[str, BinProfile] = {
    "A": BinProfile(
        bin_code="A",
        color_name="Yellow",
        bmw_category="Anatomical, Chemical, Soiled/Infectious Waste & Expired Medicines",
        plate_azimuth_deg=45.0,
        plate_tilt_deg=32.0,
        sweeper_start_deg=-60.0,
        sweeper_end_deg=65.0,
        stopper_hold_deg=85.0,
        ultrasonic_trig_pin=5,
        ultrasonic_echo_pin=6,
    ),
    "B": BinProfile(
        bin_code="B",
        color_name="Red",
        bmw_category="Contaminated Recyclable Plastic / Rubber (Gloves, Syringe Barrels, Tubing)",
        plate_azimuth_deg=135.0,
        plate_tilt_deg=30.0,
        sweeper_start_deg=-55.0,
        sweeper_end_deg=60.0,
        stopper_hold_deg=85.0,
        ultrasonic_trig_pin=13,
        ultrasonic_echo_pin=19,
    ),
    "C": BinProfile(
        bin_code="C",
        color_name="White (Translucent)",
        bmw_category="Waste Sharps including Metals (Needles, Blades, Scalpels, Syringes with Fixed Needles)",
        plate_azimuth_deg=225.0,
        plate_tilt_deg=35.0,
        sweeper_start_deg=-65.0,
        sweeper_end_deg=70.0,
        stopper_hold_deg=90.0,
        ultrasonic_trig_pin=26,
        ultrasonic_echo_pin=16,
    ),
    "D": BinProfile(
        bin_code="D",
        color_name="Blue",
        bmw_category="Glassware & Metallic Body Implants (Vials, Ampoules, Glass Slides)",
        plate_azimuth_deg=315.0,
        plate_tilt_deg=28.0,
        sweeper_start_deg=-50.0,
        sweeper_end_deg=55.0,
        stopper_hold_deg=80.0,
        ultrasonic_trig_pin=20,
        ultrasonic_echo_pin=21,
    ),
}


# --------------------------------------------------------------------------- #
# Pin Assignments (Raspberry Pi 5 BCM / ESP32-S3 Bridge)
# --------------------------------------------------------------------------- #
# Dual Flap Servos (High-torque metal-gear servos)
SWEEPER_FLAP_SERVO_PIN = 18      # Hardware PWM0
STOPPER_FLAP_SERVO_PIN = 12      # Hardware PWM1
HUB_ROTATION_SERVO_PIN = 17      # Upper robotic hub / separation-line alignment axis

# Tilting Dished Plate Stepper Drivers (TMC2209 in STEP/DIR + EN sleep mode)
TILT_X_STEP_PIN = 23
TILT_X_DIR_PIN = 24
TILT_Y_STEP_PIN = 25
TILT_Y_DIR_PIN = 8
TMC2209_ENABLE_PIN = 7           # Active LOW; pulled HIGH in ULTRA_LOW_POWER_SLEEP

# Safety, Wake-up & Auxiliary Sensors
EMERGENCY_STOP_PIN = 4           # NC (Normally Closed) button -> triggers instant halt
IR_WAKEUP_SENSOR_PIN = 14        # IR proximity sensor to wake MCU/drivers from sleep
HX711_DOUT_PIN = 22              # Load cell data pin
HX711_SCK_PIN = 27               # Load cell clock pin
EXHAUST_FAN_RELAY_PIN = 15       # HEPA/Carbon exhaust fan control


# --------------------------------------------------------------------------- #
# Flap Kinematics, Limits & Timings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlapKinematicsConfig:
    # Servo angular bounds (degrees)
    servo_min_angle: float = -90.0
    servo_max_angle: float = 90.0
    servo_min_pulse_s: float = 0.0005
    servo_max_pulse_s: float = 0.0025

    # Sweeper Flap postures
    sweeper_home_angle: float = -75.0        # Parked clear of camera FOV
    sweeper_park_angle: float = -75.0
    sweeper_sweep_speed_deg_s: float = 240.0 # Controlled sweep so fluids don't splash
    sweeper_dwell_s: float = 0.35            # Hold at chute edge to ensure item drops
    sweeper_return_s: float = 0.30

    # Stopper Flap postures (for multi-object / composite sorting)
    stopper_retracted_angle: float = 0.0     # Raised up out of the plate path
    stopper_engaged_angle: float = 85.0      # Lowered onto plate along separation line
    stopper_engage_lead_s: float = 0.25      # Lower stopper BEFORE tilting or sweeping
    stopper_release_lag_s: float = 0.20      # Lift stopper after plate returns flat

    # Dished Plate Tilt & Stepper parameters (TMC2209)
    plate_home_tilt_deg: float = 0.0
    plate_max_tilt_deg: float = 38.0
    stepper_microsteps: int = 16
    stepper_steps_per_deg: float = 8.8889    # 200 steps/rev * 16 microsteps / 360 deg
    stepper_step_pulse_us: int = 450         # Microseconds per step pulse
    plate_tilt_settle_s: float = 0.30

    # Anti-Jam Shaking Actuation (triggered when load cell > tare after sweep)
    antijam_shake_cycles: int = 4
    antijam_shake_amplitude_deg: float = 12.0
    antijam_flap_wiggle_deg: float = 20.0
    antijam_cycle_period_s: float = 0.12
    residual_weight_jam_threshold_g: float = 3.5

    # Ultra-Low-Power Sleep Management
    idle_sleep_timeout_s: float = 15.0       # Enter TMC2209 + servo sleep after 15s idle


FLAP_KINEMATICS = FlapKinematicsConfig()


# --------------------------------------------------------------------------- #
# Deterministic State Machine States (SIH2026 Architecture)
# --------------------------------------------------------------------------- #
SORTER_STATES = (
    "IDLE_SLEEP",              # Ultra-low-power sleep; awaiting IR wake-up or camera motion
    "WAKE_AND_SENSE",          # Read load cell, color sensor, ultrasonic bin levels
    "CLASSIFY_AND_COORDINATE", # CNN inference + line of separation + hazard fail-safe
    "ISOLATE_MIXED_WASTE",     # Align hub to separation line & engage Stopper Flap
    "TILT_DISHED_PLATE",       # Drive TMC2209 steppers to tilt plate toward target bin
    "SWEEP_FLAP_ACTUATE",      # Sweep Sweeper Flap across plate to push waste into bin
    "VERIFY_CLEAR_OR_SHAKE",   # Check load cell; run Anti-Jam Shaking if item stuck
    "RETURN_HOME",             # Level plate, retract Stopper & Sweeper flaps, log to PostgreSQL
    "EMERGENCY_STOP",          # Hardware interrupt lock; all PWM/steppers de-energized
)


# --------------------------------------------------------------------------- #
# Multi-Item Line of Separation & Flap Trajectory Planner
# --------------------------------------------------------------------------- #
@dataclass
class FlapActionPlan:
    target_bin: str                      # "A", "B", "C", or "D"
    target_color: str                    # "Yellow", "Red", "White (Translucent)", "Blue"
    engage_stopper: bool                 # True if mixed waste requires isolation
    hub_separation_angle_deg: float      # Angle of the line of separation (deg)
    stopper_angle_deg: float             # Stopper flap target angle
    plate_azimuth_deg: float             # Direction of dish tilt
    plate_tilt_deg: float                # Magnitude of dish tilt
    sweeper_trajectory_deg: Tuple[float, float]  # (start_angle, end_angle)
    fail_safe_override: bool = False     # True if routed via Highest-Hazard Fail-Safe
    reason: str = "standard_sort"


def compute_separation_line_angle(
    primary_box: Tuple[float, float, float, float],
    secondary_boxes: List[Tuple[float, float, float, float]],
    frame_size: Tuple[int, int] = (640, 480),
) -> float:
    """Compute the hub angle (degrees in [-90, 90]) for the Stopper Flap so it lands
    along the perpendicular bisector (line of separation) between the primary target
    waste item and the nearest secondary waste item on the dished plate.
    """
    import math

    if not secondary_boxes:
        return 0.0

    px = 0.5 * (primary_box[0] + primary_box[2])
    py = 0.5 * (primary_box[1] + primary_box[3])

    # Find nearest secondary item centroid
    best_dx, best_dy, best_dist = 0.0, 0.0, float("inf")
    for bx in secondary_boxes:
        sx = 0.5 * (bx[0] + bx[2])
        sy = 0.5 * (bx[1] + bx[3])
        d = math.hypot(sx - px, sy - py)
        if d < best_dist:
            best_dist = d
            best_dx, best_dy = sx - px, sy - py

    if best_dist < 1e-3:
        return 0.0

    # Angle of vector connecting centroids; separation line is perpendicular (+90 deg)
    conn_deg = math.degrees(math.atan2(best_dy, best_dx))
    sep_deg = conn_deg + 90.0
    # Normalise into servo range [-90, +90]
    while sep_deg > 90.0:
        sep_deg -= 180.0
    while sep_deg < -90.0:
        sep_deg += 180.0
    return round(float(sep_deg), 2)


def build_flap_action_plan(
    bin_code: str,
    mixed_waste: bool = False,
    primary_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    secondary_boxes: List[Tuple[float, float, float, float]] | None = None,
    fail_safe_override: bool = False,
    reason: str = "standard_sort",
) -> FlapActionPlan:
    """Create a deterministic kinematic action plan for the tilting plate,
    stopper flap, and sweeper flap for a given target bin (A/B/C/D).
    """
    profile = BIN_PROFILES.get(bin_code, BIN_PROFILES["C"])  # Default to highest-hazard sharps bin C
    sec_boxes = secondary_boxes or []
    engage_stopper = bool(mixed_waste or len(sec_boxes) > 0)
    sep_angle = compute_separation_line_angle(primary_box, sec_boxes) if engage_stopper else 0.0
    stopper_deg = profile.stopper_hold_deg if engage_stopper else FLAP_KINEMATICS.stopper_retracted_angle

    return FlapActionPlan(
        target_bin=profile.bin_code,
        target_color=profile.color_name,
        engage_stopper=engage_stopper,
        hub_separation_angle_deg=sep_angle,
        stopper_angle_deg=stopper_deg,
        plate_azimuth_deg=profile.plate_azimuth_deg,
        plate_tilt_deg=profile.plate_tilt_deg,
        sweeper_trajectory_deg=(profile.sweeper_start_deg, profile.sweeper_end_deg),
        fail_safe_override=fail_safe_override,
        reason=reason,
    )
