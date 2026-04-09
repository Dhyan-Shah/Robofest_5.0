from picamera2 import Picamera2
from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import time

MODEL_PATH = "best_ncnn_model"  # change to "best_ncnn" if that's your folder name
CONF_THRESHOLD = 0.40

print("Loading model...")
model = YOLO(MODEL_PATH, task="detect")
print("Model loaded")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()
print("Camera started")

app = Flask(__name__)

def generate():
    while True:
        start = time.time()

        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model(frame_bgr, imgsz=320, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        fps = 1 / (time.time() - start)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if len(results[0].boxes) > 0:
            cv2.putText(annotated, "LANDMINE DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '<html><body><img src="/video" width="640"/></body></html>'

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Stream at http://<your-pi-ip>:5000")
    print("Find your IP with: hostname -I")
    app.run(host='0.0.0.0', port=5000)