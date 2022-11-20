import time
import cv2
import numpy as np

from comms import SerialLine
from hands2d import HandDet2D

def norm(v):
    if v>1: return 1
    if v<0: return 0
    return v

iter = 0

finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
test_fing = [(8, 6), (12, 10), (16, 14), (20, 18), (4,2), (2,5), (0,5)]# pedejais merogam
thumb_Coord = (4,2)

handDet2D = HandDet2D(finger_Coord, test_fing, thumb_Coord)
serialLine = SerialLine(115200, len(test_fing)-1, 0, 1)

cap = cv2.VideoCapture(0)#ext camera
#cap = cv2.VideoCapture("/dev/video1")
print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(3), cap.get(4), cap.get(cv2.CAP_PROP_BUFFERSIZE))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#cap.set(3, 320)
#cap.set(4, 240)
avg_t = 0
try:
    while(1):
        t = time.time()
        success, image = cap.read()

        multiLandMarks = handDet2D.procImage(image)
        if multiLandMarks:
            handDet2D.handList = []
            for handLms in multiLandMarks:
                handDet2D.procHand(image, handLms)
                print(handDet2D.iterFing())
                #out = [norm(v) for v in out]
                #out = handDet2D.upDown()
                #handDet2D.calibFing()

                #print(out)
                #serialLine.compVal(out)
                #serialLine.sendVal(out)

        iter+=1

        cv2.imshow("Counting number of fingers", image)
        cv2.waitKey(1)
except KeyboardInterrupt:
    cap.release()
