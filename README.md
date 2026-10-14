# Smart Mobile Medical-Waste Collection and Segregation System (SIH 2026 — Team Pravah77)

* **Problem Statement ID:** 26115
* **Problem Statement Title:** Design and Develop a Smart Mobile Medical-Waste Collection and Segregation System
* **Theme:** MedTech / BioTech / Health Tech (PS Category: Software)
* **Team ID / Name:** 156739 — **Pravah77**
* **Hardware Target:** Raspberry Pi 5 + ESP32-S3 + OpenMV + TMC2209 Stepper Drivers + Dual Servo Flaps (Stopper & Sweeper) + HX711 Load Cell + RGB Color Sensor + 4× Ultrasonic Depth Sensors

```text
flap_config.py          kinematics, GPIO pins, separation-line maths & trajectories for Tilting Plate + Stopper & Sweeper Flaps
config.py               paths, BMW Rules 2016 4-bin mapping (A/B/C/D), hyper-parameters, sensor fusion & PostgreSQL config
dataset_prep.py         Roboflow VOC verification, BMW 4-bin balance report, leakage check, mixed-waste & glare augmentation
train.py                Model Maker EfficientDet-Lite0 training, callbacks, COCO mAP, then hands off to export
export_tflite.py        float32 / float16 / INT8 export, calibration set, size + latency + accuracy gates, deploy/ package
metrics.py              per-class & BMW 4-bin confusion matrices, precision/recall, false positives on empty-bin frames
tflite_utils.py         TF-free TFLite wrapper, specular glare filter (transparent/reflective surfaces), NMS
edge_inference.py       deterministic State Machine sorter, Dual-Flap & Tilting Plate control, sensor fusion, PostgreSQL/SQLite logging
test_pipeline_logic.py  hardware-free unit & integration test suite (16/16 passing)
