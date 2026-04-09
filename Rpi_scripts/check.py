from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30, (1280, 720))

print("Recording... press Ctrl+C to stop")
try:
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame_bgr)
except KeyboardInterrupt:
    print("Stopped")

out.release()
picam2.stop()
print("Saved test_video.mp4")