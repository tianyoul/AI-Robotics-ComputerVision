import cv2
import numpy as np


# Method from http://creat-tabu.blogspot.com/2013/08/opencv-python-hand-gesture-recognition.html

def preprocess(img):
    '''This function turns an hand image into black and white. As long as hand has darker color than the background,
    the function is able to change hand to white and background to black.'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh