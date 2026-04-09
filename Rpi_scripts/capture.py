import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    cv2.imwrite('test_image.jpg', frame)
    print("Image saved as test_image.jpg")
else:
    print("Failed to capture!")

cap.release()