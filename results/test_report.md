# AI Model Test Results

## Test Setup
- Model: best.pt
- Task: Road damage detection
- Test images: 4
- Inference mode: Image detection

## Observed Results

The trained model was tested on road-damage images.

### Successful detections
- crack1.jpg
- crack2.jpg
- crack3.jpg

### Additional test
- road_damage_test.jpg

## Important Observation

The model successfully produced bounding-box detections on the test images. However, some predictions had very low confidence and the damage classification was not consistently reliable.

## Current Limitation

The current prototype requires further validation on a larger, representative test dataset before reporting reliable precision, recall, F1-score, IoU, severity, persistence, or deterioration metrics.
