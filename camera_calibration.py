import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((5*8, 3),np.float32)
objp[:,:2] = np.mgrid[0:8, 0:5].T.reshape(-1,2)

# Arrays tp store object points and image points from all images
objpoints = [] #3d points in real world space
imgpoints = [] #2d points in image plane


def single_cam_calibration(img):
    '''
    :param img: an grayscale image with chessboard
    :return: camera matrix and distortion coefficients
    '''

    #find chessboard corners
    ret, corners = cv2.findChessboardCorners(img, (8,5), None)

    #if found, add object points, image points after refining them
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(img, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners)

    if img is not None and len(imgpoints)>0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape, None, None)
    else:
        mtx = None
        dist = None

    return mtx,dist


def single_cam_undistortion(img, mtx, dist):
    '''
    :param img: an grayscale image to be processed
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :return: an undistorted grayscale image
    '''

    if mtx is not None and dist is not None:
        h, w = img.shape
        print(img.shape)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx,dist,None,new_camera_matrix)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    else:
        dst = img
    return dst


#Test on single_cam_undistortion
vidCap = cv2.VideoCapture(0)
while True:
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    ret, img = vidCap.read()
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", img)
    mtx, dist = single_cam_calibration(img)
    newImg = single_cam_undistortion(img, mtx, dist)
    cv2.imshow("Calibrated", newImg)

cv2.destroyAllWindows()
vidCap.release()
