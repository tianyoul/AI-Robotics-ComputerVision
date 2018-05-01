import cv2
import numpy as np
from stereo_calibration import StereoCalibration
from global_variables import *

# # termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# # prepare object points
# objp = np.zeros((5*5, 3),np.float32)
# objp[:,:2] = np.mgrid[0:5, 0:5].T.reshape(-1,2)
#
# # Arrays tp store object points and image points from all images
# objpoints = [] #3d points in real world space
# imgpoints = [] #2d points in image plane


# def single_cam_calibration(img):
#     '''
#     :param img: an grayscale image with chessboard
#     :return: camera matrix and distortion coefficients
#     '''
#
#     #find chessboard corners
#     ret, corners = cv2.findChessboardCorners(img, (5,5), None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
#
#     #if found, add object points, image points after refining them
#     if ret == True:
#         print("true")
#         objpoints.append(objp)
#         cv2.cornerSubPix(img, corners, (11,11),(-1,-1), criteria)
#         imgpoints.append(corners)
#         cv2.drawChessboardCorners(img,(5,5),corners,ret)
#
#     if img is not None and len(imgpoints)>0:
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape, None, None)
#     else:
#         mtx = None
#         dist = None
#
#     return mtx,dist
#
#
# def single_cam_undistortion(img, mtx, dist):
#     '''
#     :param img: an grayscale image to be processed
#     :param mtx: camera matrix
#     :param dist: distortion coefficient
#     :return: an undistorted grayscale image
#     '''
#
#     if mtx is not None and dist is not None:
#         h, w = img.shape
#         new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#         dst = cv2.undistort(img, mtx,dist,None,new_camera_matrix)
#         x,y,w,h = roi
#         dst = dst[y:y+h, x:x+w]
#     else:
#         dst = img
#     return dst
#
#
# def two_cams_calibration(imgLeft, imgRight):
#     calibration = StereoCalibration()
#     calibrator = StereoCalibrator(9, 6, 2, (720,1080))
#
#
#     return
#
# # #Test on single_cam_undistortion
# # vidCap = cv2.VideoCapture(0)
# # while True:
# #     x = cv2.waitKey(10)
# #     char = chr(x & 0xFF)
# #     if char == 'q':
# #         break
# #     ret, img = vidCap.read()
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# #     mtx, dist = single_cam_calibration(img)
# #     cv2.imshow("Origin", img)
# #     newImg = single_cam_undistortion(img, mtx, dist)
# #     cv2.imshow("Calibrated", newImg)
# #
# # cv2.destroyAllWindows()
# # vidCap.release()


# #Test on two_cams_calibration
# vidCapLeft = cv2.VideoCapture(0)
# vidCapRight = cv2.VideoCapture(1)
# count = 0
# while True:
#     x = cv2.waitKey(10)
#     char = chr(x & 0xFF)
#     if char == 'q':
#         break
#     imgL = vidCapLeft.read()
#     imgR = vidCapRight.read()
#     # imgL = cv2.resize(imgL, (0, 0), None, .25, .25)
#     # imgR = cv2.resize(imgR, (0, 0), None, .25, .25)
#     numpy_horizontal = np.hstack((imgL, imgR))
#     cv2.imshow("Test", numpy_horizontal)


# def update_chessboard_data():
#     vidCapLeft = cv2.VideoCapture(0)
#     vidCapRight = cv2.VideoCapture(1)
#     count = 0
#     while True:
#         if count >= 10:
#             break
#         retL, imgL = vidCapLeft.read()
#         retR, imgR = vidCapRight.read()
#         x = cv2.waitKey(10)
#         char = chr(x & 0xFF)
#         if (char == 'q'):
#             count += 1
#             retL, imgL = vidCapLeft.read()
#             retR, imgR = vidCapRight.read()
#             cv2.imwrite(chessboard_dir + "/left" + str(count) + ".jpg", imgL)
#             cv2.imwrite(chessboard_dir + "/right" + str(count) + ".jpg", imgR)
#     cv2.destroyAllWindows()
#     vidCapLeft.release()
#     vidCapRight.release()



calibration = StereoCalibration('chessboard')
model = calibration.camera_model

numDisparity = 16
blockSize = 15

img_shape = (640,480)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(model['M1'], model['dist1'], model['M2'], model['dist2'],
                                                  img_shape, model['R'], model['T'])
left_map = cv2.initUndistortRectifyMap(model['M1'], model['dist1'], R1, P1, img_shape, cv2.CV_16SC2)
right_map = cv2.initUndistortRectifyMap(model['M2'], model['dist2'], R2, P2, img_shape, cv2.CV_16SC2)

vidCapLeft = cv2.VideoCapture(0)
vidCapRight = cv2.VideoCapture(1)

while True:

    retL, imgL = vidCapLeft.read()
    #img_shape = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY).shape[::-1]
    retR, imgR = vidCapRight.read()

    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if (char == 'q'):
        break

    if retL and retR:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        origImg = imgR
        cv2.imshow("left", imgL)
        cv2.imshow("right",imgR)


        newimgL = cv2.remap(imgL, left_map[0], left_map[1], 4)
        newimgR = cv2.remap(imgR, right_map[0], right_map[1], 4)
        cv2.imshow("newleft", newimgL)
        cv2.imshow("newright",newimgR)
        # # Rectify the two images taken from the two cameras
        # rectified_pair = calibration.rectify((imgL,imgR))
        # Initialize a stereo block matcher
        # x = cv2.waitKey(10)
        # char = chr(x & 0xFF)
        # if (char == 'w'):
        #     numDisparity += 16
        # elif (char == 's'):
        #     numDisparity -= 16
        # elif (char == 'a'):
        #     blockSize -= 2
        # elif (char == 'd'):
        #     blockSize += 2
        block_matcher = cv2.StereoBM_create(numDisparity, blockSize)

        # Compute a disparity image

        disparity = block_matcher.compute(imgL,imgR)
        # # Show normalized version of image
        # cv2.imshow("Left", rectified_pair[0])
        # cv2.imshow("Right", rectified_pair[1])
        cv2.imshow("camera", disparity/255.)

cv2.destroyAllWindows()
vidCapRight.release()
vidCapLeft.release()