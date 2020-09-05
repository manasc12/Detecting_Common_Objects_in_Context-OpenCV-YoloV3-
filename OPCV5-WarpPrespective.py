import cv2
import numpy as np

width, height= 200, 250
img=cv2.imread("Resource/cat.jpg")

pt1=np.float32([[123,29],[547,22],[141,472],[484,484]])
pt2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pt1,pt2)
imgConv=cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Image",img)
cv2.imshow("Warp Img",imgConv)
cv2.waitKey(0)