from picamera2 import Picamera2
from flask import Flask, Response
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import time

MODEL_PATH = "best_float32.tflite"
CONF_THRESHOLD = 0.40
INPUT_SIZE = 320

print("Loading model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (INPUT_SIZE, INPUT_SIZE)}))
picam2.start()
print("Camera started")

app = Flask(__name__)

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, frame):
    h, w = frame.shape[:2]
    predictions = np.squeeze(output[0])

    boxes = []
    confidences = []

    for pred in predictions.T:
        confidence = pred[4]
        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        x1 = int((cx - bw / 2) * w / INPUT_SIZE)
        y1 = int((cy - bh / 2) * h / INPUT_SIZE)
        x2 = int((cx + bw / 2) * w / INPUT_SIZE)
        y2 = int((cy + bh / 2) * h / INPUT_SIZE)

        boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            [[x, y, x2-x, y2-y] for x, y, x2, y2 in boxes],
            confidences, CONF_THRESHOLD, 0.4
        )
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Landmine {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def generate():
    while True:
        start = time.time()
        frame = picam2.capture_array()

        img_input = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        frame = postprocess(output, frame)

        fps = 1.0 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Landmine Detector</title></head>
    <body style="background:black; text-align:center">
        <h1 style="color:white">Landmine Detection</h1>
        <img src="/video" width="640">
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting server at http://<rpi-ip>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)