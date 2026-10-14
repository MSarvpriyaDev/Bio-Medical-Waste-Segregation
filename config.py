"""Central configuration for the Smart Mobile Medical-Waste Collection and
Segregation System (SIH 2026 - PS ID 26115 - Team Pravah77).

Every script imports this module; nothing else hard-codes paths, class names,
BMW Rules 2016 4-bin mappings, sensor thresholds, or actuator hyper-parameters.
Edit CLASS_NAMES first - it must match the Roboflow Pascal-VOC export exactly
(run ``python dataset_prep.py discover`` to see what the XML files use).

Compatible with Python 3.9 (training env) and 3.11 (Raspberry Pi 5 OS Bookworm).
"""
from __future__ import annotations

import os
from pathlib import Path

import flap_config as fcfg

# --------------------------------------------------------------------------- #
# Project Metadata (SIH 2026 - Team Pravah77)
# --------------------------------------------------------------------------- #
PROBLEM_STATEMENT_ID = "26115"
TEAM_ID = "156739"
TEAM_NAME = "Pravah77"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("WASTE_DATA_DIR", PROJECT_ROOT / "data"))

# Roboflow "Pascal VOC" export, unzipped: train/ valid/ test/, each holding
# image files and same-named .xml files side by side.
RAW_DIR = DATA_DIR / "roboflow_voc"
RAW_SPLITS = {
    "train": RAW_DIR / "train",
    "val": RAW_DIR / "valid",
    "test": RAW_DIR / "test",
}

# Sanitised copies (+ offline augmentation for train) written by dataset_prep.py.
PREPARED_DIR = DATA_DIR / "prepared"
PREPARED_SPLITS = {name: PREPARED_DIR / name for name in ("train", "val", "test")}

# Optional photos of EMPTY bins / liners (no waste in view). Used for
# copy-paste augmentation and (separately!) for false-positive testing.
BACKGROUND_DIR = DATA_DIR / "backgrounds"   # used to synthesise training images
NEGATIVES_DIR = DATA_DIR / "negatives"      # held out, NEVER used for training

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model_dir"          # Model Maker checkpoints
EXPORT_DIR = ARTIFACTS_DIR / "export"            # .tflite variants + deploy/
CALIBRATION_DIR = ARTIFACTS_DIR / "calibration"  # INT8 representative images
REPORT_DIR = ARTIFACTS_DIR / "reports"           # JSON metrics / dataset report

DEPLOY_MODEL_NAME = "model.tflite"
DEPLOY_LABELS_NAME = "labels.txt"
DEPLOY_MANIFEST_NAME = "manifest.json"

SEED = 42

# --------------------------------------------------------------------------- #
# Classes & BMW Rules 2016 4-Bin Mapping (Type A / B / C / D)
# --------------------------------------------------------------------------- #
# Order defines class ids: id 0 == first entry.
CLASS_NAMES = ["gloves", "mask", "sharps", "syringe", "vial"]
NUM_CLASSES = len(CLASS_NAMES)

# Biomedical Waste Management Rules 2016 (4 Colour-Coded Bins):
#   Type A (Yellow) : Anatomical waste, chemical waste, soiled/infectious dressings, masks, expired medicines
#   Type B (Red)    : Contaminated recyclable plastics/rubber (gloves, syringe barrels, catheters, IV sets)
#   Type C (White)  : Waste sharps including metals (needles, blades, scalpels, fixed-needle syringes)
#   Type D (Blue)   : Glassware & metallic body implants (vials, ampoules, broken glass slides)
BMW_BINS = ("A", "B", "C", "D")
BIN_COLOR_MAP = {code: prof.color_name for code, prof in fcfg.BIN_PROFILES.items()}
BIN_CATEGORY_DESCRIPTION = {code: prof.bmw_category for code, prof in fcfg.BIN_PROFILES.items()}

CLASS_TO_BIN = {
    "mask": "A",        # Yellow Bin (Type A: Soiled / Infectious PPE)
    "gloves": "B",      # Red Bin    (Type B: Contaminated Recyclable Plastic/Latex)
    "syringe": "B",     # Red Bin    (Type B: Plastic Syringe; overridden to C if needle/sharp co-detected)
    "sharps": "C",      # White Bin  (Type C: Needles, Scalpels, Blades)
    "vial": "D",        # Blue Bin   (Type D: Glass Vials & Ampoules)
}

# Lower index = higher hazard priority when multiple / composite classes appear together.
# Highest Hazard Fail-Safe: Sharps (C) > Infectious/Anatomical (A) > Glassware (D) > Recyclable Plastic (B).
CLASS_PRIORITY = ["sharps", "mask", "vial", "syringe", "gloves"]
BIN_HAZARD_PRIORITY = ["C", "A", "D", "B"]

# Highest Hazard fail-safe priority when ambiguous composite items or low-confidence detections occur
FALLBACK_BIN = "C"              # Ambiguous / composite hazard -> White Sharps / Highest-Hazard Bin
AMBIGUOUS_CONFIDENCE_FLOOR = 0.48  # Detections in [0.48, TRIGGER_CONFIDENCE) with load-cell weight trigger fail-safe

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
MODEL_NAME = "efficientdet_lite0"
INPUT_SIZE = 320            # EfficientDet-Lite0 is fixed at 320x320
MAX_DETECTIONS = 25         # NMS output slots baked into the .tflite graph

BATCH_SIZE = 16
EPOCHS = 60
TRAIN_WHOLE_MODEL = True    # False = train head only (faster, less accurate)

# Original EfficientDet recipe: lr 0.08 at batch 64 -> scale linearly.
BASE_LEARNING_RATE = 0.08
LR_REFERENCE_BATCH = 64
LR_WARMUP_EPOCHS = 1.0
LR_WARMUP_FRACTION = 0.1    # warm-up starts at 10 % of the target lr
LEARNING_RATE = BASE_LEARNING_RATE * BATCH_SIZE / LR_REFERENCE_BATCH

EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-3

# --------------------------------------------------------------------------- #
# Dataset preparation / augmentation
# --------------------------------------------------------------------------- #
MIN_BOX_PX = 4                  # boxes smaller than this (either side) are dropped
NEAR_DUP_HAMMING = 4            # dHash distance <= this counts as a near-duplicate

# Repeat-factor sampling (LVIS style) for class imbalance:
RFS_THRESHOLD = 0.30
AUG_BASE_COPIES = 1
AUG_MAX_COPIES = 5

# Synthetic copy-paste images built on empty-bin backgrounds (needs BACKGROUND_DIR).
N_COPY_PASTE_IMAGES = 300

# Camera is fixed and looks down on the hydrophobic non-stick plate -> flips and
# 90-degree rotations are physically valid. Includes optical reflection / glare
# robustness for transparent & reflective surfaces (vials, syringes, wet bio-fluids).
AUGMENTATION = {
    # geometric
    "p_hflip": 0.5, "p_vflip": 0.5, "p_rot90": 0.5,
    "p_affine": 0.7, "rotate_deg": 20.0, "scale": (0.85, 1.15), "translate": 0.08,
    "min_visible": 0.5,
    # photometric (tuned for optical distortion on transparent/reflective surfaces)
    "p_brightness_contrast": 0.8, "brightness": 0.25, "contrast": 0.25,
    "p_gamma": 0.4, "gamma": (0.7, 1.5),
    "p_hsv": 0.5, "hue": 4, "sat": 0.25, "val": 0.20,
    "p_color_cast": 0.4, "cast": 0.08,
    "p_gradient": 0.5, "p_shadow": 0.3, "p_glare": 0.35,
    "p_motion_blur": 0.15, "p_gauss_blur": 0.15,
    "p_noise": 0.3, "p_jpeg": 0.3,
    # background-clutter robustness
    "p_clutter": 0.4, "max_clutter": 4,
    # copy-paste of extra objects onto real training frames (mixed-waste 4-type scenes)
    "p_paste_on_sample": 0.30,
}

# --------------------------------------------------------------------------- #
# Export / quantisation
# --------------------------------------------------------------------------- #
CALIBRATION_IMAGES = 300        # INT8 representative dataset size (100-500 is typical)
EXPORT_VARIANTS = ("float32", "float16", "int8")
MODEL_SIZE_BUDGET_MB = 8.0      # flag deploy candidates above this
MAX_QUANT_AP_DROP = 0.03        # allowed absolute AP drop (INT8 vs float32)
EVAL_SCORE_THRESHOLDS = (0.3, 0.5, 0.7)   # operating points for P/R tables
EVAL_PRIMARY_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Edge inference (Raspberry Pi 5 + ESP32-S3 + OpenMV Co-processor)
# --------------------------------------------------------------------------- #
NUM_THREADS = 4                 # Raspberry Pi 5 has 4 Cortex-A76 cores
CAMERA_SIZE = (640, 480)        # capture size (the model input is resized from this)
CAMERA_FPS = 30
CAMERA_ROI = None
# Polarising / anti-reflection camera filter + locked exposure to eliminate reflections
PICAMERA_CONTROLS = {}
POLARIZING_GLARE_FILTER = True  # Software specular highlight suppression before CNN inference

DISPLAY_SCORE_THRESHOLD = 0.40  # boxes drawn on screen
TRIGGER_CONFIDENCE = 0.70       # minimum confidence that may fire the sorter directly
NMS_IOU = 0.50                  # extra class-aware NMS on top of the in-graph NMS
AGNOSTIC_NMS_IOU = 0.60         # collapse boxes of different classes on one object
STABLE_FRAMES = 5               # consecutive frames a class must persist
COOLDOWN_S = 3.0                # lockout after a sort command

# --------------------------------------------------------------------------- #
# Hardware Sensors: Load Cell (HX711), Color Sensor, Ultrasonic Depth, Safety
# --------------------------------------------------------------------------- #
# Plausible weight ranges (grams) per class for Load Cell (HX711) sensor fusion
CLASS_WEIGHT_RANGE_G = {
    "sharps": (0.3, 35.0),
    "syringe": (2.0, 65.0),
    "vial": (5.0, 180.0),
    "gloves": (3.0, 50.0),
    "mask": (1.5, 25.0),
}
LOAD_CELL_MIN_TRIGGER_G = 0.5   # Minimum weight on plate to confirm physical item presence
LOAD_CELL_MAX_SAFE_G = 1500.0   # Overload threshold -> triggers anti-jam / alert

# Color Sensor (TCS34725 RGB) dominant hue fallback when camera fails or is occluded
COLOR_SENSOR_TO_BIN = {
    "yellow": "A",
    "red": "B",
    "white": "C",
    "blue": "D",
}

# Ultrasonic Depth Sensor (HC-SR04) per bin for volume tracking
BIN_DEPTH_CM = 45.0             # Empty bin distance from top ultrasonic sensor to bottom
BIN_FULL_ALERT_PERCENT = 85.0   # Alert & lock bin when fill volume exceeds 85%

# --------------------------------------------------------------------------- #
# Sorting Mechanism & Flap Actuation (see flap_config.py for full profiles)
# --------------------------------------------------------------------------- #
GPIO_ENABLED = True             # False -> dry-run logging only
GPIO_PIN_FACTORY = None         # None (gpiozero default) | "pigpio" (steadier PWM)
SERVO_PIN = fcfg.SWEEPER_FLAP_SERVO_PIN
STOPPER_SERVO_PIN = fcfg.STOPPER_FLAP_SERVO_PIN
HUB_SERVO_PIN = fcfg.HUB_ROTATION_SERVO_PIN
SERVO_MIN_ANGLE = fcfg.FLAP_KINEMATICS.servo_min_angle
SERVO_MAX_ANGLE = fcfg.FLAP_KINEMATICS.servo_max_angle

# Azimuth / Servo target angle per BMW 2016 bin (A: Yellow, B: Red, C: White, D: Blue)
BIN_SERVO_ANGLE = {
    "A": -60.0,   # Yellow Bin (Type A)
    "B": -20.0,   # Red Bin    (Type B)
    "C": 25.0,    # White Bin  (Type C)
    "D": 65.0,    # Blue Bin   (Type D)
}
SERVO_SETTLE_S = 0.6

# --------------------------------------------------------------------------- #
# Digital Traceability Database (PostgreSQL + Local SQLite Fallback)
# --------------------------------------------------------------------------- #
POSTGRES_DSN = os.environ.get(
    "BMW_POSTGRES_DSN",
    "postgresql://pravah77:sih2026@localhost:5432/biomedical_waste_db",
)
SQLITE_FALLBACK_PATH = ARTIFACTS_DIR / "waste_traceability.sqlite3"
