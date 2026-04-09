import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "test_image.jpg"
CONF_THRESHOLD = 0.40
INPUT_SIZE = 320

print("Loading model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frame = cv2.imread(IMAGE_PATH)
h, w = frame.shape[:2]

img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

predictions = np.squeeze(output[0])
boxes = []
confidences = []

for i in range(predictions.shape[1]):
    confidence = predictions[4][i]
    if confidence < CONF_THRESHOLD:
        continue
    cx = predictions[0][i]
    cy = predictions[1][i]
    bw = predictions[2][i]
    bh = predictions[3][i]

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))

    boxes.append([x1, y1, x2, y2])
    confidences.append(float(confidence))

if len(boxes) > 0:
    indices = cv2.dnn.NMSBoxes(
        [[x, y, x2-x, y2-y] for x, y, x2, y2 in boxes],
        confidences, CONF_THRESHOLD, 0.4
    )
    print(f"{len(indices)} landmine(s) detected!")
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        conf = confidences[i]
        print(f"  - Confidence: {conf:.2f} at [{x1},{y1},{x2},{y2}]")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Landmine {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
else:
    print("No landmines detected!")

cv2.imwrite('result.jpg', frame)
print("Result saved as result.jpg")