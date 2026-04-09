"""
Real-time Gesture Recognition
Run: python gesture_recognition.py
"""

import cv2
import numpy as np
import onnxruntime as ort

# ── CONFIG ───────────────────────────────────────────
MODEL_PATH = r"C:\Users\Dell\Desktop\Robofest 5.0\Codes\Models\Gesture_Recognition\best_model.onnx"
INPUT_SIZE = (224, 224)

# Labels loaded from your classes.json
LABELS = {
    0: "fist",
    1: "idle",
    2: "open_palm",
    3: "thumb_up",
    4: "two_fingers"
}
# ─────────────────────────────────────────────────────

# Load model
print("Loading model...")
session    = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print(f"✓ Model loaded | Input: {session.get_inputs()[0].shape}")
print(f"✓ Classes: {list(LABELS.values())}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Could not open webcam")
    exit()

print("✓ Camera opened | Press Q to quit\n")

# Colors per gesture
COLORS = {
    0: (0,   200, 100),   # fist        — green
    1: (120, 120, 120),   # idle        — gray
    2: (0,   220, 255),   # open_palm   — cyan
    3: (0,   180, 255),   # thumb_up    — orange-ish
    4: (180,  80, 255),   # two_fingers — purple
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Preprocess
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, INPUT_SIZE).astype(np.float32) / 255.0
    tensor  = np.expand_dims(resized, axis=0)

    # Inference
    outputs   = session.run(None, {input_name: tensor})
    probs     = outputs[0][0]
    top_idx   = int(np.argmax(probs))
    top_conf  = probs[top_idx] * 100
    top_label = LABELS[top_idx]
    color     = COLORS.get(top_idx, (0, 255, 220))

    # Terminal output
    bar = "█" * int(top_conf / 5) + "░" * (20 - int(top_conf / 5))
    print(f"\r  {top_label:<14} {bar} {top_conf:5.1f}%  ", end="", flush=True)

    # ── Top bar ──
    cv2.rectangle(frame, (0, 0), (w, 65), (10, 10, 10), -1)

    # Gesture name
    cv2.putText(frame, top_label.upper(), (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

    # Confidence %
    conf_text = f"{top_conf:.1f}%"
    cv2.putText(frame, conf_text, (w - 110, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int((w - 20) * top_conf / 100)
    cv2.rectangle(frame, (10, 54), (w - 10, 62), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 54), (10 + bar_w, 62), color, -1)

    # ── Bottom class bars ──
    num = len(LABELS)
    slot_w = w // num
    for idx, lbl in LABELS.items():
        prob  = probs[idx] if idx < len(probs) else 0
        x     = idx * slot_w
        bw    = slot_w - 3
        bh    = int(prob * 90)
        c     = COLORS.get(idx, (100, 100, 100))

        # Highlight top class
        if idx == top_idx:
            cv2.rectangle(frame, (x, h - bh - 2), (x + bw, h), c, -1)
        else:
            cv2.rectangle(frame, (x, h - bh), (x + bw, h),
                          (c[0]//3, c[1]//3, c[2]//3), -1)

        cv2.putText(frame, lbl, (x + 3, h - bh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255) if idx == top_idx else (150, 150, 150),
                    1, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100:.0f}%", (x + 3, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition  |  Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n\nStopped.")