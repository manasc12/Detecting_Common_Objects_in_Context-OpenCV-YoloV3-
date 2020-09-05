import cv2
import numpy as np

frameHeight=640
frameWidth=480

cap = cv2.VideoCapture(0)
cap.set(3,frameHeight)
cap.set(4,frameWidth)
cap.set(10,150)
myColors=[[23,83,168,52,212,255],
          [43,121,120,89,235,255]] #Set these values in myColors from colorPicker.py

drawColor=[[0,255,255],
           [0,255,0]] #which color u wanna draw on canvas

myPoints=[] #empty list to capture all PolyDP points to draw

def findColor(imgHSV,color,name,drawColor):
    lower=np.array(color[0:3])
    upper=np.array(color[3:6])
    mask=cv2.inRange(imgHSV,lower,upper)
    getContours(mask,drawColor)
    #cv2.imshow(name,mask)

def getContours(mask,drawColor):
    contours, hierarchy= cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area>300): #REDUCING the unwanted noises
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            arcLen=cv2.arcLength(cnt,True)
            polyDP=cv2.approxPolyDP(cnt,0.02*arcLen,True)
            polyDPLen=len(polyDP)
            x,y,w,h=cv2.boundingRect(polyDP)
            myPoints.append([x+w//2,y,drawColor])
            cv2.circle(imgContour, (x+w//2, y), 10, drawColor, cv2.FILLED)


while True:
    succ, img= cap.read()
    imgContour= img.copy()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    i=0
    for color in myColors:
        name="VideoHSV--"+str(i+1)
        findColor(imgHSV,color,name,drawColor[i])
        i = i + 1
    for point in myPoints:
        cv2.circle(imgContour,(point[0],point[1]),7,point[2],cv2.FILLED)

    cv2.imshow("Video-Image", imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
