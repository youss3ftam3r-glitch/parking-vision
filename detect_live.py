"""
Parking Spot Detection — LIVE LOOP
------------------------------------
Continuously reads frames from a camera (webcam for now, Pi camera later),
runs the PRETRAINED YOLOv8n model on each frame, checks spot occupancy,
and POSTs to your dashboard server ONLY when a spot's state changes.

SETUP:
    pip install ultralytics opencv-python requests

BEFORE RUNNING:
    1. Make sure your dashboard server (server.js) is running
       (node server.js, from the parking-prototype folder)
    2. Update SERVER_URL below if the server isn't on localhost
    3. Paste your SPOTS dict from spot_marker.html below (use a frame
       grabbed from your webcam this time, not the old photo — camera
       angle/resolution needs to match what you're now capturing live)

USAGE:
    python detect_live.py

    - A window will pop up showing the live feed with boxes/spots drawn
    - Press 'q' to quit
    - Press 's' to save the current frame as "calibration_frame.jpg" —
      useful for re-running spot_marker.html to redefine SPOTS if your
      camera position changes
"""

from ultralytics import YOLO
import cv2
import requests
import time
import numpy as np

# ---- CONFIG ----
CAMERA_INDEX = 1  # 0 = default webcam. Try 1 if you have multiple cameras.
SERVER_URL = "http://localhost:3000/api/update"
CHECK_INTERVAL_SECONDS = 2  # how often to run detection
CONFIRM_READS = 2  # consecutive matching reads needed before flipping state (debounce)
OVERLAP_THRESHOLD = 0.3  # fraction of car box that must be inside spot polygon

# Paste your SPOTS dict from spot_marker.html here (must match your
# CURRENT camera's frame resolution/angle, re-mark if you moved the camera)
SPOTS = {
    "spot1": [(1, 658), (149, 564), (286, 587), (101, 684)],
    "spot2": [(105, 690), (293, 586), (421, 591), (260, 693)],
    "spot3": [(268, 694), (423, 591), (538, 598), (422, 702)],
    "spot4": [(429, 703), (542, 599), (693, 604), (590, 707)],
    "spot5": [(598, 709), (695, 606), (830, 609), (768, 712)],
    "spot6": [(770, 717), (834, 611), (972, 617), (941, 722)],
    "spot7": [(945, 727), (978, 617), (1116, 625), (1122, 734)],
    "spot8": [(1127, 734), (1123, 625), (1269, 632), (1306, 743)],
    "spot9": [(1312, 744), (1274, 634), (1433, 640), (1507, 758)],
    "spot10": [(1513, 758), (1442, 640), (1588, 647), (1707, 767)],
}
# -----------------

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck


def polygon_mask(shape, polygon):
    mask = np.zeros(shape[:2], dtype="uint8")
    pts = np.array(polygon, dtype="int32")
    cv2.fillPoly(mask, [pts], 255)
    return mask


def box_overlap_fraction(box, spot_mask):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, spot_mask.shape[1]), min(y2, spot_mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_area = (x2 - x1) * (y2 - y1)
    overlap = spot_mask[y1:y2, x1:x2]
    overlap_area = (overlap > 0).sum()
    return overlap_area / box_area if box_area > 0 else 0.0


def send_update(spot_id, occupied):
    try:
        r = requests.post(
            SERVER_URL,
            json={"spotId": spot_id, "occupied": occupied},
            timeout=3,
        )
        print(f"  -> POST {spot_id}={occupied} : {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  -> POST FAILED for {spot_id}: {e}")


def main():
    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: couldn't open camera. Try changing CAMERA_INDEX to 1.")
        return

    # precompute spot masks once (they don't change frame to frame)
    ret, sample_frame = cap.read()
    if not ret:
        print("ERROR: couldn't read a frame from the camera.")
        return
    spot_masks = {sid: polygon_mask(sample_frame.shape, poly) for sid, poly in SPOTS.items()}

    last_reported = {sid: None for sid in SPOTS}  # None = not yet reported
    pending_state = {sid: None for sid in SPOTS}
    pending_count = {sid: 0 for sid in SPOTS}

    print("Running. Press 'q' in the video window to quit, 's' to save a calibration frame.\n")

    last_check = 0
    last_results_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, retrying...")
            time.sleep(0.5)
            continue

        now = time.time()
        if now - last_check >= CHECK_INTERVAL_SECONDS:
            last_check = now
            results = model(frame, classes=VEHICLE_CLASSES, verbose=False)
            last_results_boxes = [b.xyxy[0].tolist() for b in results[0].boxes]

            for spot_id, mask in spot_masks.items():
                occupied_now = any(
                    box_overlap_fraction(box, mask) >= OVERLAP_THRESHOLD
                    for box in last_results_boxes
                )

                if occupied_now == pending_state[spot_id]:
                    pending_count[spot_id] += 1
                else:
                    pending_state[spot_id] = occupied_now
                    pending_count[spot_id] = 1

                if pending_count[spot_id] >= CONFIRM_READS:
                    if occupied_now != last_reported[spot_id]:
                        last_reported[spot_id] = occupied_now
                        print(f"[{time.strftime('%H:%M:%S')}] {spot_id}: {'OCCUPIED' if occupied_now else 'EMPTY'}")
                        send_update(spot_id, occupied_now)

        # draw overlay every frame (fast, no model call)
        display = frame.copy()
        for box in last_results_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for spot_id, polygon in SPOTS.items():
            occupied = last_reported.get(spot_id)
            color = (0, 0, 255) if occupied else (0, 255, 0) if occupied is False else (128, 128, 128)
            pts = np.array(polygon, dtype="int32")
            cv2.polylines(display, [pts], True, color, 2)
            cx = int(sum(p[0] for p in polygon) / len(polygon))
            cy = int(sum(p[1] for p in polygon) / len(polygon))
            label = "OCCUPIED" if occupied else "EMPTY" if occupied is False else "..."
            cv2.putText(display, f"{spot_id}: {label}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Parking Detection (press q to quit, s to save frame)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("calibration_frame.jpg", frame)
            print("Saved calibration_frame.jpg — use this with spot_marker.html to redefine SPOTS.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
