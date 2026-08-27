# InfraSight

## AI-Based Road Damage Detection & Inspection

InfraSight is a computer-vision prototype that detects visible road damage from road images using a trained YOLO model and produces visual detection evidence.

## Problem

Manual road inspection is time-consuming and difficult to scale. InfraSight aims to support faster first-level road inspection by automatically identifying visible road defects from imagery.

## Current Working Prototype

The submitted prototype currently demonstrates:

- Road image input
- AI-based road damage detection
- Damage-class prediction
- Confidence score
- Bounding-box visualization
- Generated detection output
- Local application/API for inference

## AI Model

Model: YOLO-based object detection

Trained model:

`model/best.pt`

### Damage Classes

- D00 — Longitudinal crack
- D10 — Transverse crack
- D20 — Alligator crack
- D40 — Pothole
- D50 — Other damage

## System Flow

Road Image
↓
YOLO Detection
↓
Damage Class + Confidence
↓
Bounding Box
↓
Detection Result

## Repository Structure

```text
InfraSight/
├── README.md
├── main.py
├── requirements.txt
├── index.html
├── camera.html
│
├── model/
│   └── best.pt
│
└── results/
    └── test_report.md
