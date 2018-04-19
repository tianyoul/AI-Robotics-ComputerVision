import numpy as np
import cv2
from camera_calibration import *
from matplotlib import pyplot as plt

def undistort(img):
    '''Undistort image. Camera calibration'''
    # copy parameters to arrays
    K = np.array([[1755.04324, 0., 650.63], [0, 1754.95349, 545.9389], [0, 0, 1]])
    d = np.array([.16858, 0.57600, 0, 0, 0])  # just use first two terms

    h,w = img.shape[:2]
    newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
    newImg = cv2.undistort(img, K, d, None, newcamera)

    return newImg

vidCapLeft = cv2.VideoCapture(0)
vidCapRight = cv2.VideoCapture(1)

while True:
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    retL, imgL = vidCapLeft.read()
    retR, imgR = vidCapRight.read()
    cv2.imshow("L", imgL)
    (w,h) = imgL.shape[:2]
    imgR = imgR[:w, :h]
    cv2.imshow("R", imgR)

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # #Python 2.7
    # stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=16, SADWindowSize=15)
    # disparity = stereo.compute(imgL, imgR, disptype=cv2.CV_32F)

    #Python 3.6
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)

    norm_coeff = 255 / disparity.max()
    cv2.imshow("disparity", disparity * norm_coeff / 255)
    # plt.ion()
    # plt.imshow(disparity, 'gray')
    # plt.show()


cv2.destroyAllWindows()
vidCapLeft.release()
vidCapRight.release()