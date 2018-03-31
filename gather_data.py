import cv2
import numpy as np
import os
import time


vidCap = cv2.VideoCapture(0)

# variables

# The square area (500 * 500)
top_left = (50, 100)
bottom_right = (550, 600)  # openCV order
green = (100, 180, 50)

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# directory
dir_prefix = "img/"
label_folder = ['fist', 'hand', 'peace', 'none']


data_to_load = ['test'] #choose which data to load

i = 0
while i < len(data_to_load):
    x = cv2.waitKey(10)
    ret, vd_img = vidCap.read()
    cv2.rectangle(vd_img, top_left, bottom_right, green)
    cv2.putText(vd_img, "Press C to capture " + label_folder[i] + " data", (5, 30), font, 1, (255, 255, 255), 3)
    cv2.imshow("background", vd_img)

    char = chr(x & 0xFF)
    if char == 'c':
        folder_path = dir_prefix + data_to_load[i]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for j in range(50):
            time.sleep(0.1)
            ret, vd_img = vidCap.read()
            cv2.rectangle(vd_img, top_left, bottom_right, green)
            cv2.putText(vd_img, "Progress: " + str(j), (5, 30), font, 1, (255, 255, 255), 3)
            #Currently not displaying progress
            cv2.imshow("background", vd_img)
            area_img = vd_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv2.imwrite(folder_path + "/" + data_to_load[i] + str(j) + ".jpg", area_img)

        i += 1 # load next gesture


cv2.destroyAllWindows()
vidCap.release()