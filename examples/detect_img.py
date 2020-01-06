import sys
import numpy as np
import cv2

img = cv2.imread('circles.png', 1)

lower_range = np.array([110, 100, 100], dtype=np.uint8)
upper_range = np.array([130, 255, 255], dtype=np.uint8)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('mask', mask)
cv2.imshow('image', img)

while (1):
    k = cv2.waitKey(0)
    if (k == 27):
        break

cv2.destroyAllWindows()
