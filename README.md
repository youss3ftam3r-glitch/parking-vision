# Parking Vision

A work-in-progress parking occupancy prototype using a pretrained YOLOv8n model, OpenCV, and a camera feed. I am building this project while learning Python and computer vision.

## What it does

- Detects cars, buses, and trucks in camera frames.
- Compares vehicle bounding boxes with manually marked parking-space polygons.
- Requires two consecutive matching readings before reporting an occupancy change.
- Displays vehicle boxes and color-coded parking spaces in a live window.
- Sends occupancy changes to a configurable HTTP endpoint.

This uses pretrained model weights; it is not a model trained from scratch. Occupancy is a bounding-box overlap heuristic, not a validated production parking system.

## Setup

Use Python 3.10 or newer in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The detector loads `yolov8n.pt`. Ultralytics can download these pretrained weights on first use. Model weights and captured camera frames are excluded from this repository.

## Mark parking spaces

1. Set `CAMERA_INDEX` in `detect_live.py` for your camera (currently `1`; try `0` for the default camera).
2. Run `python detect_live.py` and press `s` to save a calibration frame, then `q` to quit.
3. Open `spot_marker.html` in a browser and choose the calibration image.
4. Click four corners for each parking space, then copy the generated Python dictionary.
5. Replace `SPOTS` in `detect_live.py` with that dictionary and restart the detector.

The included polygon coordinates belong to an earlier camera setup. Recalibrate whenever the camera angle or image resolution changes.

## Run

```sh
python detect_live.py
```

- `q`: quit.
- `s`: save `calibration_frame.jpg` locally.
- Gray outline: waiting for a confirmed reading.
- Green outline: empty.
- Red outline: occupied.

Detection runs every two seconds by default. Adjust `CHECK_INTERVAL_SECONDS`, `CONFIRM_READS`, and `OVERLAP_THRESHOLD` in the configuration section as needed.

## Optional dashboard integration

`SERVER_URL` defaults to `http://localhost:3000/api/update`. The expected request is:

```json
{"spotId": "spot1", "occupied": true}
```

A dashboard/backend is **not included** in this snapshot. Detection and the camera overlay can run without it, but update requests will log connection failures. Point `SERVER_URL` to your own endpoint to receive changes.

## Current limitations

- Requires a fixed camera and manually defined parking spaces.
- Accuracy depends on lighting, occlusion, camera placement, and vehicle detection.
- Failed HTTP updates are logged, but are not retried until another state change.
- Camera operation and full inference have not been verified as part of preparing this repository.

## Tools

Python, [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), OpenCV, NumPy, and Requests. Third-party libraries and model weights retain their respective licenses.
