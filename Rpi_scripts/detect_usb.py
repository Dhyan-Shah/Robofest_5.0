from flask import Flask, Response, jsonify
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import time
import base64

MODEL_PATH = "best_float32.tflite"
CONF_THRESHOLD = 0.40
INPUT_SIZE = 320

print("Loading model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded")

cap = cv2.VideoCapture(0)
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

    count = 0
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            [[x, y, x2-x, y2-y] for x, y, x2, y2 in boxes],
            confidences, CONF_THRESHOLD, 0.4
        )
        for i in indices:
            count += 1
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Landmine {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame, count

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Failed to capture frame'})

    result_frame, count = postprocess(
        interpreter.get_tensor(output_details[0]['index']),
        frame.copy()
    )

    # Run inference
    img_input = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    result_frame, count = postprocess(
        interpreter.get_tensor(output_details[0]['index']),
        frame.copy()
    )

    _, buffer = cv2.imencode('.jpg', result_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': img_base64,
        'count': count,
        'message': f'{count} landmine(s) detected!' if count > 0 else 'No landmines detected'
    })

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Landmine Detector</title>
        <style>
            body { background: #111; color: white; text-align: center; font-family: Arial; }
            h1 { color: #ff4444; }
            button {
                background: #ff4444; color: white; border: none;
                padding: 15px 40px; font-size: 18px; border-radius: 8px;
                cursor: pointer; margin: 10px;
            }
            button:hover { background: #cc0000; }
            #result-img { max-width: 640px; border: 3px solid #ff4444; border-radius: 8px; }
            #message { font-size: 22px; margin: 15px; color: #ffcc00; }
        </style>
    </head>
    <body>
        <h1>ðŸš¨ Landmine Detector</h1>
        <img src="/video" width="640"><br>
        <button onclick="capture()">ðŸ“¸ Capture & Detect</button>
        <div id="message"></div>
        <img id="result-img" style="display:none">

        <script>
            function capture() {
                document.getElementById('message').innerText = 'Detecting...';
                fetch('/capture')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('message').innerText = data.message;
                        const img = document.getElementById('result-img');
                        img.src = 'data:image/jpeg;base64,' + data.image;
                        img.style.display = 'block';
                    });
            }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting server at http://10.76.144.96:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)