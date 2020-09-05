import cv2

path="Resource/haarcascade_frontalface_default.xml"
haarClass=cv2.CascadeClassifier(path)

img=cv2.imread("Resource/lena.png")
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

boundingRect=haarClass.detectMultiScale(imgGray,1.1,4)


for (x,y,w,h) in boundingRect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


cv2.imshow("Image-Lena",cv2.resize(img,(400,400)))
cv2.waitKey(0)