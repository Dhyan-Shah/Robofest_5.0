"""
Landmine Detection — YOLOv8 TFLite
Run: python mine_detection.py
"""

import cv2
import numpy as np
import tensorflow as tf

# ── CONFIG ───────────────────────────────────────────
MODEL_PATH  = r"C:\Users\Dell\Desktop\Robofest 5.0\Codes\Models\Mine_Detection\best_saved_model\best_float32.tflite"  # path to your .tflite model
LABELS      = ["landmines"]
CONF_THRESH = 0.35            # confidence threshold
IOU_THRESH  = 0.45            # NMS IoU threshold
INPUT_SIZE  = 640             # YOLOv8 default input size
# ─────────────────────────────────────────────────────

# Load TFLite model
print("Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Detect input size from model
model_h = input_details[0]['shape'][1]
model_w = input_details[0]['shape'][2]
INPUT_SIZE = model_w
print(f"✓ Model loaded | Input: {model_h}x{model_w}")
print(f"✓ Class: {LABELS[0]}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Could not open webcam")
    exit()

print("✓ Camera opened | Press Q to quit\n")

# Colors
BOX_COLOR     = (0, 60, 255)    # red-ish for danger
TEXT_BG_COLOR = (0, 40, 200)
TEXT_COLOR    = (255, 255, 255)
ALERT_COLOR   = (0, 0, 255)

def preprocess(frame, size):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, orig_w, orig_h, conf_thresh, iou_thresh):
    """
    YOLOv8 TFLite output shape: [1, 5, num_anchors] (cx,cy,w,h,conf)
    or [1, num_anchors, 5] depending on export — handle both.
    """
    pred = output[0]  # remove batch dim

    # Transpose if needed: expect [num_anchors, 5+]
    if pred.shape[0] < pred.shape[-1]:
        pred = pred.T  # [5, anchors] → [anchors, 5]

    boxes, scores = [], []

    for det in pred:
        # det: [cx, cy, w, h, conf, (class scores if multi-class)]
        if len(det) == 4 + len(LABELS):
            cx, cy, w, h = det[0], det[1], det[2], det[3]
            confs = det[4:]
            conf  = float(np.max(confs))
        elif len(det) == 5:
            cx, cy, w, h, conf = det
        else:
            continue

        if conf < conf_thresh:
            continue

        # Convert from normalized [0,1] to pixel coords
        x1 = int((cx - w / 2) * orig_w)
        y1 = int((cy - h / 2) * orig_h)
        x2 = int((cx + w / 2) * orig_w)
        y2 = int((cy + h / 2) * orig_h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)

    # NMS
    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            results.append((x, y, x + w, y + h, scores[i]))
    return results

frame_count = 0
import time
fps_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    orig_h, orig_w = frame.shape[:2]

    # Preprocess
    inp = preprocess(frame, INPUT_SIZE)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess
    detections = postprocess(output, orig_w, orig_h, CONF_THRESH, IOU_THRESH)

    # ── Draw detections ──
    mine_found = len(detections) > 0

    for (x1, y1, x2, y2, conf) in detections:
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Corner accents
        corner = 12
        cv2.line(frame, (x1, y1), (x1 + corner, y1), ALERT_COLOR, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), ALERT_COLOR, 3)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), ALERT_COLOR, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), ALERT_COLOR, 3)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), ALERT_COLOR, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), ALERT_COLOR, 3)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), ALERT_COLOR, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), ALERT_COLOR, 3)

        # Label
        label = f"LANDMINE {conf*100:.1f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), TEXT_BG_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

    # ── Top status bar ──
    cv2.rectangle(frame, (0, 0), (orig_w, 50), (10, 10, 10), -1)

    if mine_found:
        status     = f"⚠  LANDMINE DETECTED  ({len(detections)})"
        status_col = (0, 60, 255)
    else:
        status     = "SCANNING..."
        status_col = (0, 200, 100)

    cv2.putText(frame, status, (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_col, 2, cv2.LINE_AA)

    # FPS
    frame_count += 1
    if frame_count % 15 == 0:
        fps = 15 / (time.time() - fps_time)
        fps_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (orig_w - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # Terminal
    print(f"\r  {'⚠ MINE DETECTED' if mine_found else 'scanning...':30s}  detections: {len(detections)}  FPS: {fps:.1f}  ",
          end="", flush=True)

    cv2.imshow("Landmine Detection  |  Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n\nStopped.")