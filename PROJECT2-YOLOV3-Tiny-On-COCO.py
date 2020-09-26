import cv2
import numpy as np

frameHeight=450
frameWidth=450

cap = cv2.VideoCapture('/Users/manasc12/PycharmProjects/OpencvPython/Resource/VID_20200822_165642.mp4')
#cap.set(3,frameHeight)
#cap.set(4,frameWidth)
#cap.set(cv2.CAP_PROP_FPS, 25)
#cap.set(10,250)

classFile='Resource/coco.names'
classNames=[]
with open(classFile,'rt') as f:
    classNames= f.read().rstrip('\n').split('\n')

print(len(classNames))
#print(classNames)

modelConf="yolov3-tiny.cfg"
modelWeight="yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(modelConf,modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def detBbox(outputs,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confidences=[]

    for output in outputs:
        for line in output:
            scores=np.array(line)[5:]
            confidence_idx=np.argmax(scores)
            if scores[confidence_idx]>0.5:
                w,h=int(line[2]*wT),int(line[3]*hT)
                x,y=int(line[0]*wT-w/2),int(line[1]*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(confidence_idx)
                confidences.append(float(scores[confidence_idx]))
    indices=cv2.dnn.NMSBoxes(bbox,confidences,0.5,0.3)
    for i in indices:
    #for counter in range(len(bbox)):
        counter=i[0]
        x,y,w,h=bbox[counter][0],bbox[counter][1],bbox[counter][2],bbox[counter][3]
        cv2.rectangle(img,(x,y),(x+w,y+h),[0,((confidences[counter]-0.5)*1020),(1-confidences[counter])*1020],2)
        cv2.rectangle(img,(x,y+h-20),(x+w,y+h),[0,((confidences[counter]-0.5)*1020),(1-confidences[counter])*1020],cv2.FILLED)
        cv2.putText(img, classNames[classIds[counter]]+" - "+str(int(confidences[counter]*100))+"%", (x + 20, y+h-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    return img

size=(int(cap.get(3)), int(cap.get(4)))
result = cv2.VideoWriter('newvideo.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
while True:
    succ, img=cap.read()

    blob=cv2.dnn.blobFromImage(img,1/255,size=(320,320),mean=[0,0,0],crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    #print(layerNames)
    #print(net.getUnconnectedOutLayers())  ## to get the outputlayers from all the layers it gives their (index+1)
    outPutLayers= [[(i[0]-1),layerNames[i[0]-1]] for i in net.getUnconnectedOutLayers()]
    #print(np.array(outPutLayers))

    outputs=net.forward(np.array(outPutLayers)[:,1])
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)


    img=detBbox(outputs,img)
    result.write(img)
    cv2.imshow("Video",img)
    if cv2.waitKey(1)==ord('q'):
        cap.release()
        result.release()
        cv2.destroyAllWindows()
        print("The video was successfully saved")
        break
cap.release()
result.release()
cv2.destroyAllWindows()
print("The video was successfully saved")

