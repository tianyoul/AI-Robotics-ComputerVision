import cv2
import numpy as np


'''This file tests uses the preprocessing technique in a realtime setting.'''

cap = cv2.VideoCapture(0)  # creating camera object

while (cap.isOpened()):
    ret1, img = cap.read()  # reading the frames
    #img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret2, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('drawing', thresh1)


    k = cv2.waitKey(10)
    char = chr(k & 0xFF)
    if char == 'q':
        break


cv2.destroyAllWindows()
cap.release()
