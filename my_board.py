import cv2
import numpy as np
import numpy as np
import os
import HandModule as hm

path="images"
myList=os.listdir(path)
#print(myList)
drawcolor=(250,0,250)
upList =[]
brush=15
erase=50

for i in myList:
    img=cv2.imread(f'{path}/{i}')
    upList.append(img)

head=upList[0]
myCanvas=np.zeros((720,1280,3),np.uint8)

cap =cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
xp,yp=0,0

detector =hm.handDetector(detectionCon=0.85)


while True:
    success, img=cap.read()
    img=cv2.flip(img,1)
    
    img=detector.findHands(img)
    lmlist=detector.findPosition(img)

    if len(lmlist)!=0:
        print(lmlist)

        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        fingers=detector.finger()

        if fingers[1] and fingers[2]:
            if y1<125:
                if 250<x1<450:
                    head=upList[0]
                    drawcolor=(250,0,250)
                elif 550<x1<750:
                    head=upList[1]
                    drawcolor=(250,0,0)
                elif 800<x1<950:
                    head=upList[2]
                    drawcolor=(0,250,0)
                elif 1050<x1<1200:
                    head=upList[3]
                    drawcolor=(0,0,0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawcolor, cv2.FILLED)
            xp,yp=x1,y1

        if fingers[1] and fingers[2]==0:
            cv2.circle(img,(x1,y1),15,drawcolor,cv2.FILLED)

            if xp==0 and yp==0:
                xp=x1
                yp=y1

            if drawcolor==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, erase)
                cv2.line(myCanvas, (xp, yp), (x1, y1), drawcolor, erase)
            cv2.line(img,(xp,yp),(x1,y1),drawcolor,brush)
            cv2.line(myCanvas, (xp, yp), (x1, y1), drawcolor, brush)
            xp,yp=x1,y1

    imggray=cv2.cvtColor(myCanvas,cv2.COLOR_BGR2GRAY)
    _,imga=cv2.threshold(imggray,50,255,cv2.THRESH_BINARY_INV)
    imga=cv2.cvtColor(imga,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imga)
    img=cv2.bitwise_or(img,myCanvas)

    #img[0:125,0:1267]=head

    #img=cv2.addWeighted(img,0.5,myCanvas,0.5,0)
    cv2.imshow("Image",img)
    #cv2.imshow("canvas",myCanvas)
    cv2.waitKey(1)
