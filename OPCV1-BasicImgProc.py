import cv2
import numpy as np

img=cv2.imread("Resource/cat.jpg")
kernel=np.ones((5,5),np.uint8)

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny=cv2.Canny(img,150,200)
imgDilation=cv2.dilate(imgCanny,kernel,iterations=1)
imgEroded=cv2.erode(imgDilation,kernel,iterations=1)



cv2.imshow("Original Img",img)
cv2.imshow("Gray Img",imgGray)
cv2.imshow("Blurr Img",imgBlur)
cv2.imshow("Canny Img",imgCanny)
cv2.imshow("Dilated Img",imgDilation)
cv2.imshow("Eroded Img",imgEroded)

cv2.waitKey(0)