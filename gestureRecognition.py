import cv2
import numpy as np


vidCap = cv2.VideoCapture(0)

top_left = (50, 100)
bottom_right = (500, 600)  # In opencv, need to be reversed in np
green = (100, 180, 50)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    x = cv2.waitKey(10)
    ret, bgimg = vidCap.read()
    cv2.rectangle(bgimg, top_left, bottom_right, green)
    cv2.putText(bgimg, "Press C to capture background", (5, 30), font, 1, (255, 255, 255), 3)
    cv2.imshow("background", bgimg)

    bg_area = bgimg[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    char = chr(x & 0xFF)
    if char == 'c':
        cv2.imwrite("background.jpg", bg_area)
        cv2.destroyAllWindows()
        break


while True:
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    ret, img = vidCap.read()

    cv2.rectangle(img, top_left, bottom_right, green)
    cv2.imshow("cam", img)

    gesture = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    diff_img = cv2.absdiff(bg_area, gesture) * 2 #Multiply by 2 to intensify the color of hand
    #cv2.imshow("gesture", diff_img)

    edges = cv2.Canny(diff_img, 100, 200)
    cv2.imshow("gesture", edges)

cv2.destroyAllWindows()
vidCap.release()



