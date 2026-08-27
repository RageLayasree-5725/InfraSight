# InfraSight

## AI-Based Road Damage Detection

InfraSight is a computer-vision prototype that detects road damage
from images using a trained YOLO model and produces visual
bounding-box evidence for detected defects.

## Problem

Manual road inspection is time-consuming and difficult to scale.
InfraSight provides an automated first-level inspection from road images.

## Current Working Features

- Road-damage image upload
- AI inference using trained YOLO model
- Bounding-box visualization
- Damage-class prediction
- Confidence score
- Example test images
- Reproducible local inference

## Supported Damage Classes

- D00 — Longitudinal crack
- D10 — Transverse crack
- D20 — Alligator crack
- D40 — Pothole
- D50 — Other damage

## Architecture

Image
  ↓
YOLO model
  ↓
Object detection
  ↓
Class + confidence + bounding box
  ↓
Visual result

## How to Run

### Install dependencies

pip install -r requirements.txt

### Run application

python main.py

Then open the displayed/local application URL.

## Model

The trained model is stored in:

model/best.pt

## Testing

Test images and detection outputs are available in:

examples/
results/

See results/test_report.md for the measured results.

## Limitations

The current prototype is an image-based detection system.
Detection accuracy depends on image quality, viewpoint,
lighting and the training data distribution.

## Future Scope

- Larger and more diverse road dataset
- Improved confidence calibration
- Temporal persistence across repeated inspections
- Automated maintenance prioritization
- Deployment on drone/edge hardware
