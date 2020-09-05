import cv2
import numpy as np

img= cv2.imread("Resource/cat.jpg")
imgResized=cv2.resize(img,(100,100))
imgCropped=img[0:200,300:500]
cv2.imshow("Original",img)
cv2.imshow("Resized",imgResized)
cv2.imshow("cropped",imgCropped)
cv2.waitKey(0)