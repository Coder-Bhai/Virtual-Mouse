import pyautogui
import hands_module
import cv2
import time
import math

cap = cv2.VideoCapture(0)

pTime = 0
#increase the confidence to increase precision
detector = hands_module.handDetect(detectionCnf=0.9)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPos(img, draw=False)
    if len(lmlist):
        x1, y1 = lmlist[4][1], lmlist[4][2]  #position of THUMB_TIP
        x2, y2 = lmlist[8][1], lmlist[8][2]  #position of INDEX_FINGER_TIP
        cx, cy = (x1+x2)//2, (y1+y2)//2      #centre of line joining above two pts
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 6)
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        pyautogui.moveTo(cx * 2, cy * 2)
        if length <= 20:
            pyautogui.leftClick()

    cTime = time.time()
    if cTime != pTime:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0
    pTime = cTime
    #displaying fps
    cv2.putText(img, f'FPS:{int(fps)}', (40, 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)
    #displaying final image
    cv2.imshow('WebCam', img)
    cv2.waitKey(1)
