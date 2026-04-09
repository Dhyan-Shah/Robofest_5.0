from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()

frame = picam2.capture_array()
cv2.imwrite("test_frame.jpg", frame)
print("Saved test_frame.jpg")

picam2.stop()