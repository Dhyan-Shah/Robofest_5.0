"""
Gesture Recognition Server
--------------------------
Runs inference using ONNX Runtime (no internet needed).
Supports .onnx models converted from Keras/TF.

Usage:
  pip install flask onnxruntime numpy opencv-python
  python app.py
Then open: http://localhost:5000
"""

import os, io, base64, json, time
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Global model state ──
model_session = None
model_labels  = []
input_shape   = (224, 224)
input_name    = None

# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ─────────────────────────────────────────────
@app.route('/load_model', methods=['POST'])
def load_model():
    global model_session, model_labels, input_shape, input_name

    onnx_file = request.files.get('model')
    labels_raw = request.form.get('labels', '')

    if not onnx_file:
        return jsonify({'error': 'No model file provided'}), 400

    try:
        import onnxruntime as ort
        model_bytes = onnx_file.read()
        model_session = ort.InferenceSession(model_bytes)

        # Detect input shape
        inp = model_session.get_inputs()[0]
        input_name = inp.name
        shape = inp.shape  # e.g. [None, 224, 224, 3]
        if len(shape) == 4 and shape[1] and shape[2]:
            h = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 224
            w = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 224
            input_shape = (h, w)

        # Detect output classes
        out = model_session.get_outputs()[0]
        num_classes = out.shape[-1]

        # Parse labels
        model_labels = [l.strip() for l in labels_raw.split('\n') if l.strip()]
        if not model_labels:
            model_labels = [f'Class {i}' for i in range(num_classes)]

        return jsonify({
            'success': True,
            'input_shape': list(input_shape),
            'num_classes': num_classes,
            'labels': model_labels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model_session is None:
        return jsonify({'error': 'No model loaded'}), 400

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        # Decode base64 image from webcam
        img_b64 = data['image'].split(',')[-1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess
        h, w = input_shape
        resized = cv2.resize(frame, (w, h)).astype(np.float32) / 255.0
        tensor  = np.expand_dims(resized, axis=0)

        # Infer
        t0 = time.perf_counter()
        outputs = model_session.run(None, {input_name: tensor})
        ms = round((time.perf_counter() - t0) * 1000, 1)

        probs = outputs[0][0].tolist()
        top_idx = int(np.argmax(probs))

        return jsonify({
            'probs':    probs,
            'top_idx':  top_idx,
            'top_label': model_labels[top_idx] if top_idx < len(model_labels) else f'Class {top_idx}',
            'top_conf': round(probs[top_idx] * 100, 1),
            'infer_ms': ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🤙  Gesture Recognition Server")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
