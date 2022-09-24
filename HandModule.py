import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.detectionCon = detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(static_image_mode=self.mode,max_num_hands=self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
        self.mpDraw= mp.solutions.drawing_utils
        self.tipids=[4,8,12,16,20]

    def findHands(self,img, draw=False):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img,handNo=0, draw=False):
        self.lmList=[]

        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

        return self.lmList

    def finger(self):
        fingers=[]

        if self.lmList[self.tipids[0]][1] < self.lmList[self.tipids[0]-1][1] :
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1,5):
            if self.lmList[self.tipids[i]][2] < self.lmList[self.tipids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    cap = cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Can not open camera")
            break;
        img=detector.findHands(img)
        mylist =detector.findPosition(img)
        #if len(mylist)!=0:
            #print(mylist[4])
        cv2.imshow("Image",img)
        cv2.waitKey(1)



if __name__=="__main__":
    main()