import numpy as np
import cv2
import glob
from numpy import array

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.
imgpoints2 = []

leftimages = glob.glob('chessboard_img/left*.ppm')
rightimages = glob.glob('chessboard_img/right*.ppm')

for fname in leftimages:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints1.append(corners)

for fname in rightimages:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints2.append(corners)


cv2.stereoCalibrate()

vidCapLeft = cv2.VideoCapture(0)
vidCapRight = cv2.VideoCapture(1)
while True:
    retL, imgL = vidCapLeft.read()
    # (1080, 1920)
    imgL = cv2.resize(imgL, (1280, 720))
    #cv2.imshow("L", imgL)
    retR, imgR = vidCapRight.read()
    # (720, 1280) This is the webcam of macbook pro
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if (char == 'q'):
        break

    rectified_left = cv2.remap(imgL, H1, None, 0)
    rectified_right = cv2.remap(imgR, H2, None, 0)
    cv2.imshow("Left",rectified_left)
    cv2.imshow("Right", rectified_right)
    # Initialize a stereo block matcher
    block_matcher = cv2.StereoBM_create(16, 15)
    # Compute a disparity image
    disparity = block_matcher.compute(cv2.cvtColor(rectified_left,cv2.COLOR_BGR2GRAY),cv2.cvtColor(rectified_right,cv2.COLOR_BGR2GRAY))
    # # Show normalized version of image
    # cv2.imshow("Left", rectified_pair[0])
    # cv2.imshow("Right", rectified_pair[1])
    cv2.imshow("camera", disparity/255.)

