import cv2

vidCap = cv2.VideoCapture(0)

top_left = (50, 100)
bottom_right = (500, 600)  # In opencv, need to be reversed in np
green = (100, 180, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
index = 0
while True:

    x = cv2.waitKey(10)
    ret, bgimg = vidCap.read()
    cv2.rectangle(bgimg, top_left, bottom_right, green)
    cv2.putText(bgimg, "Press C to capture image", (5, 30), font, 1, (255, 255, 255), 3)
    cv2.imshow("background", bgimg)

    bg_area = bgimg[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    char = chr(x & 0xFF)
    if char == 'c':
        gray_bg = cv2.cvtColor(bg_area, cv2.COLOR_BGR2GRAY)
        image_name = "gesture" + str(index) + ".jpg"
        cv2.imwrite(image_name, bg_area)
        index += 1

    if char == 'q':
        break


cv2.destroyAllWindows()
vidCap.release()