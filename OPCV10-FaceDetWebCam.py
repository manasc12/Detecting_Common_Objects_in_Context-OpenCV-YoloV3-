import cv2

frameHeight=640
frameWidth=500
path="Resource/haarcascade_frontalface_default.xml"
haarClass=cv2.CascadeClassifier(path)

cap=cv2.VideoCapture(0)
cap.set(3,frameHeight)
cap.set(4,frameWidth)
cap.set(10,150)


while True:
    succ, img= cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boundingRect = haarClass.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in boundingRect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






