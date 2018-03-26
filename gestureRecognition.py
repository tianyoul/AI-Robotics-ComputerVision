import cv2
import numpy as np


vidCap = cv2.VideoCapture(0)

top_left = (50, 100)
bottom_right = (500, 600)  # In opencv, need to be reversed in np
green = (100, 180, 50)
font = cv2.FONT_HERSHEY_SIMPLEX

# while True:
#     x = cv2.waitKey(10)
#     ret, bgimg = vidCap.read()
#     cv2.rectangle(bgimg, top_left, bottom_right, green)
#     cv2.putText(bgimg, "Press C to capture background", (5, 30), font, 1, (255, 255, 255), 3)
#     cv2.imshow("background", bgimg)
#
#     bg_area = bgimg[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
#     char = chr(x & 0xFF)
#     if char == 'c':
#         gray_bg = cv2.cvtColor(bg_area, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite("background.jpg", bg_area)
#         cv2.destroyAllWindows()
#         break
#
#
# while True:
#     x = cv2.waitKey(10)
#     char = chr(x & 0xFF)
#     if char == 'q':
#         break
#     ret, img = vidCap.read()
#
#     cv2.rectangle(img, top_left, bottom_right, green)
#     cv2.imshow("cam", img)
#
#     gesture = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
#     gray_gesture = cv2.cvtColor(gesture, cv2.COLOR_BGR2GRAY)
#
#     #diff_img = cv2.absdiff(bg_area, gesture) * 2 #Multiply by 2 to intensify the color of hand
#     #cv2.imshow("gesture", diff_img)
#
#     diff_img = cv2.absdiff(gray_bg, gray_gesture) * 2
#
#     edges = cv2.Canny(diff_img, 100, 200)
#     cv2.imshow("gesture", edges)

while True:
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    ret, img = vidCap.read()

    cv2.rectangle(img, top_left, bottom_right, green)
    gesture = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    gray_gesture = cv2.cvtColor(gesture, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_gesture, (5,5), 0)

    #Otsu segmentation algorithm
    _, thresh = cv2.threshold(gray_gesture, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    #Finding the biggest contour
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    max_area = 0
    for contour in contours:
        if cv2.contourArea(contour) > max_area:
            max_area = cv2.contourArea(contour)
            max_contour = contour
    #hull = cv2.convexHull(max_contour)
    cv2.drawContours(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], [max_contour], 0, (0, 0, 255), 2)
    cv2.imshow("Camera", img)

cv2.destroyAllWindows()
vidCap.release()



