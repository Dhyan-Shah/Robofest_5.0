from picamera2 import Picamera2
import time
import os

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(2)  # warm up camera

def get_next_filename(base="test", ext=".jpg"):
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

while True:
    input("Press Enter to capture (or Ctrl+C to quit)...")
    filename = get_next_filename()
    cam.capture_file(filename)
    print(f"Image saved as {filename}")

cam.stop()