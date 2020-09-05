import cv2

frameHeight=640
frameWidth=480

cap = cv2.VideoCapture(0)
cap.set(3,frameHeight)
cap.set(4,frameWidth)
cap.set(10,150)


while True:
    succ, img= cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
