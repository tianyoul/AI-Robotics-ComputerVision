import cv2

vidCapLeft = cv2.VideoCapture(0)
vidCapRight = cv2.VideoCapture(1)
count = 1

while True:
    if count > 10:
        break
    retL, imgL = vidCapLeft.read()
    retR, imgR = vidCapRight.read()
    cv2.imshow("Left", imgL)
    cv2.imshow("Right", imgR)
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if (char == 'q'):
        retL, imgL = vidCapLeft.read()
        retR, imgR = vidCapRight.read()
        cv2.imwrite("test_images/left_" + str(count).zfill(2)  + ".ppm", imgL)
        cv2.imwrite("test_images/right_" + str(count).zfill(2)  + ".ppm", imgR)
        count += 1
        print(count)
cv2.destroyAllWindows()
vidCapLeft.release()
vidCapRight.release()