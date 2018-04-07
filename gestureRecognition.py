import cv2
import numpy as np


vidCap = cv2.VideoCapture(0)

top_left = (50, 100)
bottom_right = (500, 600)  # In opencv, need to be reversed in np

green = (100, 180, 50)
font = cv2.FONT_HERSHEY_SIMPLEX

# HSV value that we consider to be skin color 0,45,45 22,255,255
lower_threshold = np.array([0, 45, 30], dtype="uint8")
upper_threshold = np.array([22, 255, 255], dtype="uint8")

rectangle_area = (((300,315),(315,330)),((300,400),(315,415)), ((350, 370),(365,385)), ((250, 450),(265,465))
                  ,((230,355),(245,370)))
avg_colors = []
# while True:
#     ret, img = vidCap.read()
#     img = cv2.flip(img, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     cv2.putText(img, "Press C to continue", (5, 30), font, 1, (255, 255, 255), 3)
#     cv2.putText(img, "Cover rectangles with palm", (5, 100), font, 1, (255, 255, 255), 3)
#     for i in range(len(rectangle_area)):
#         cv2.rectangle(img, rectangle_area[i][0], rectangle_area[i][1], (255, 0, 255),2)
#     x = cv2.waitKey(10)
#     char = chr(x & 0xFF)
#     if char == 'c':
#         for i in range(len(rectangle_area)):
#             avg_color_per_row = np.average(
#                 img[rectangle_area[i][0][1]:rectangle_area[i][1][1], rectangle_area[i][0][0]:rectangle_area[i][1][1]],
#                 axis=0)
#             avg_color = np.average(avg_color_per_row, axis=0)
#             avg_colors[i] = avg_color
#         avg_color_H = sorted(avg_colors[0])
#         avg_color_S = sorted(avg_colors[1])
#         avg_color_V = sorted(avg_colors[2])
#
#         lower_threshold = np.array([avg_color_H[0],avg_color_S[0],avg_color_V[0]], dtype="uint8")
#         upper_threshold = np.array
#         break
#     cv2.imshow("Camera",img)

while True:
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    ret, img = vidCap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.rectangle(img, top_left, bottom_right, green)
    gesture = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    skin_mask = cv2.inRange(gesture, lower_threshold, upper_threshold)

    # apply a series of erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations= 4)

    #Finding the biggest contour
    _, contours, _ = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        contours = sorted(contours, key=cv2.contourArea)
        cv2.drawContours(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], [contours[0]], 0, (0, 0, 255), 2)

    cv2.imshow("s", skin_mask)
    #cv2.imshow("Camera", img)

cv2.destroyAllWindows()
vidCap.release()


