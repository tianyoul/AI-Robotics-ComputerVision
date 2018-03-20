import cv2

# Capture video stream
vidCap = cv2.VideoCapture(0)
while True:
    ret, img = vidCap.read()
    cv2.imshow("Webcam", img)
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    if char == 's':
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        cv2.imwrite("capture.jpg", image)

cv2.destroyAllWindows()
vidCap.release()
