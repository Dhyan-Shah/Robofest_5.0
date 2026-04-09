# Gesture Recognition — Local Python Server

No internet required. Runs inference server-side with ONNX Runtime.

---

## 1. Install dependencies

```bash
pip install flask onnxruntime numpy opencv-python
```

---

## 2. Convert your Keras .h5 model to ONNX (one time only)

```bash
pip install tf2onnx tensorflow
python -m tf2onnx.convert --keras model.h5 --output model.onnx
```

Or if you have a SavedModel folder:
```bash
python -m tf2onnx.convert --saved-model ./saved_model_dir --output model.onnx
```

---

## 3. Run the server

```bash
cd gesture_app
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 4. Use the app

1. Click **Choose model.onnx** and select your converted model
2. Type your class labels (one per line) in the text box
3. Click **⬆ Load Model**
4. Click **▶ Start Camera** — live inference begins!

---

## Troubleshooting

- **Camera not working?** Make sure you open via `http://localhost:5000`, not by double-clicking the HTML file (browsers block webcam on `file://`)
- **Wrong predictions?** Check that your class labels are in the same order as during training
- **Model load error?** Make sure the .onnx was converted from the same model architecture (MobileNetV2)
