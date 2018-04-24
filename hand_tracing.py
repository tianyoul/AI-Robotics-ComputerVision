import cv2
import math

# The square area (500 * 500)
top_left = (50, 100)
bottom_right = (550, 600)  # openCV order
green = (100, 180, 50)

# Default setting for center of the contour and color of hand
cX, cY = -1, -1 # the coordinates in the whole image (including 50 and 100 offset on x and y directions)
handColor = -1

vidCap = cv2.VideoCapture(0)
while True:
    ret, img = vidCap.read()
    cv2.putText(img, "Make sure your hand occupies as much space in the box as possible.", (5, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 200, 200), 3)
    cv2.putText(img, "Press c to confirm and proceed", (5, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 200, 200), 3)
    cv2.rectangle(img, top_left, bottom_right, green)
    cv2.imshow("Camera",img)
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'c':
        # Decide the center of contour area
        # Since we already know that hand occupies as much as space as possible.. Then its color must be the dominant color.
        area_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        imgray = cv2.cvtColor(area_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 70, 255, 1)
        thresh = cv2.blur(thresh, (5, 5),0)

        im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)

        # Calculate the center of the biggest contour
        M = cv2.moments(contours[-1])
        cX = int(M["m10"] / M["m00"]) + 50
        cY = int(M["m01"] / M["m00"]) + 100
        handColor = imgray[cX-50,cY-100]
        break


while True:
    ret, img = vidCap.read()
    cv2.putText(img, "Press q to quit program", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,200,200), 3)
    cv2.rectangle(img, top_left, bottom_right, green)

    area_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    imgray = cv2.cvtColor(area_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, handColor + 15, 255, 0)
    cv2.imshow("test", thresh)

    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    print(len(contours))

    # Calculate the center of the biggest contour
    if (len(contours) != 0):
        newM = cv2.moments(contours[-1])
        newcX = int(newM["m10"] / newM["m00"]) + 50
        newcY = int(newM["m01"] / newM["m00"]) + 100

    if (math.hypot(newcX-cX, newcY-cY) > 10):
        # if the new center is too far away, it means lighting condition changes and we need to recalculate the color to seperate out the hand
        handColor = imgray[cX-50, cY-100]
    else:
        cX = newcX
        cY = newcY
        cv2.circle(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], (cX, cY), 7, (255, 255, 255), -1)
        cv2.drawContours(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], [contours[-1]], 0, (0, 0, 255), 2)

    cv2.imshow("Camera", img)

    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break

vidCap.release()
cv2.destroyAllWindows()
