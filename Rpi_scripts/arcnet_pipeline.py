import cv2
import os
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = r"C:\Users\Dell\Desktop\Robofest 5.0\Codes\Models\Mine_Detection\best_saved_model\best_float32.tflite"
INPUT_FOLDER   = r"C:/Users/Dell/Desktop/final"
OUTPUT_FOLDER  = "result"
CONF_THRESHOLD = 0.40
NMS_THRESHOLD  = 0.40
INPUT_SIZE     = 320
IMAGE_EXTS     = (".jpg", ".jpeg", ".png", ".bmp")
TILE_OVERLAP   = 0.25    # 25% overlap between tiles
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Letterbox ─────────────────────────────────────────────────────────────────
def letterbox(img, target=320, color=(114, 114, 114)):
    h, w  = img.shape[:2]
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target - new_w
    pad_h = target - new_h
    top   = pad_h // 2;  bottom = pad_h - top
    left  = pad_w // 2;  right  = pad_w - left
    img   = cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    return img, scale, left, top

# ── Infer on one tile ─────────────────────────────────────────────────────────
def infer_tile(tile):
    th, tw = tile.shape[:2]
    img, scale, pad_left, pad_top = letterbox(tile, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output      = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(output[0])

    boxes, confidences = [], []
    for i in range(predictions.shape[1]):
        conf = predictions[4][i]
        if conf < CONF_THRESHOLD:
            continue
        cx = predictions[0][i]; cy = predictions[1][i]
        bw = predictions[2][i]; bh = predictions[3][i]

        x1 = int((cx - bw / 2 - pad_left) / scale)
        y1 = int((cy - bh / 2 - pad_top)  / scale)
        x2 = int((cx + bw / 2 - pad_left) / scale)
        y2 = int((cy + bh / 2 - pad_top)  / scale)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(tw, x2), min(th, y2)

        boxes.append([x1, y1, x2, y2])
        confidences.append(float(conf))

    return boxes, confidences

# ── Sliding window tiling ─────────────────────────────────────────────────────
def detect_with_tiling(frame):
    H, W = frame.shape[:2]

    # Each tile = half the image = 2x zoom for the model
    tile_w = W // 2
    tile_h = H // 2
    step_x = int(tile_w * (1 - TILE_OVERLAP))
    step_y = int(tile_h * (1 - TILE_OVERLAP))

    all_boxes, all_confs = [], []

    # Build tile offsets, always include right/bottom edge
    xs = list(range(0, W - tile_w + 1, step_x))
    ys = list(range(0, H - tile_h + 1, step_y))
    if not xs or xs[-1] + tile_w < W: xs.append(W - tile_w)
    if not ys or ys[-1] + tile_h < H: ys.append(H - tile_h)

    for ty in ys:
        for tx in xs:
            tile = frame[ty:ty + tile_h, tx:tx + tile_w]
            boxes, confs = infer_tile(tile)
            # Shift coords back to full-image space
            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                all_boxes.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty])
                all_confs.append(conf)

    # Full-image pass to catch large/far objects
    full_boxes, full_confs = infer_tile(frame)
    all_boxes.extend(full_boxes)
    all_confs.extend(full_confs)

    if not all_boxes:
        return [], []

    # Global NMS to remove duplicates across tiles
    nms_input = [[x, y, x2 - x, y2 - y] for x, y, x2, y2 in all_boxes]
    indices   = cv2.dnn.NMSBoxes(nms_input, all_confs, CONF_THRESHOLD, NMS_THRESHOLD)
    final_boxes = [all_boxes[i] for i in indices]
    final_confs = [all_confs[i] for i in indices]
    return final_boxes, final_confs

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded\n")

# ── Collect images ────────────────────────────────────────────────────────────
if not os.path.exists(INPUT_FOLDER):
    print(f"[ERR] Folder not found: {INPUT_FOLDER}")
    exit()

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTS
])

if not image_files:
    print(f"[ERR] No images found in: {INPUT_FOLDER}")
    exit()

print(f"Found {len(image_files)} image(s) in {INPUT_FOLDER}")
print("-" * 50)

# ── Process each image ────────────────────────────────────────────────────────
for fname in image_files:
    IMAGE_PATH = os.path.join(INPUT_FOLDER, fname)
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"[SKIP] Could not read: {fname}")
        continue

    boxes, confs = detect_with_tiling(frame)

    if boxes:
        print(f"[!] {fname} -> {len(boxes)} landmine(s) detected!")
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            print(f"     - conf={conf:.2f}  bbox=[{x1},{y1},{x2},{y2}]")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Landmine {conf:.2f}",
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        print(f"[ok] {fname} -> No landmines detected.")

    name     = os.path.splitext(fname)[0]
    out_path = os.path.join(OUTPUT_FOLDER, name + "_result.jpg")
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

print("-" * 50)
print(f"All results saved in: {OUTPUT_FOLDER}/")