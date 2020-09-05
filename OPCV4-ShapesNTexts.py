import cv2
import numpy as np

img=np.zeros((512,512,3),np.uint8)
img[:,:]=255,100,86

cv2.line(img,(0,0),(img.shape[0],img.shape[1]),color=(0,0,255),thickness=2)
cv2.rectangle(img,(0,0),(img.shape[0]-200,50),(0,255,200),cv2.FILLED)
cv2.circle(img,(400,300),30,(130,140,200),cv2.FILLED)


cv2.putText(img,"My Text",(350,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(220,180,120),thickness=2)
print(img)
cv2.imshow("Img",img)
cv2.waitKey(0)