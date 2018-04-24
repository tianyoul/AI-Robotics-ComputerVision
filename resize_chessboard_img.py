import cv2
import glob

frames = glob.glob("chessboard_img/left*.ppm")
count = 1
for fname in frames:
    img = cv2.imread(fname)
    img = cv2.resize(img, (1280, 720))
    cv2.imwrite("chessboard_img/left_" + str(count).zfill(2) + ".ppm", img)
    count += 1

frames = glob.glob("chessboard_img/*.ppm")
for fname in frames:
    img = cv2.imread(fname)
    print(img.shape[:-1])