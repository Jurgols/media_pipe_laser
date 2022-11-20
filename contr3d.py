import time
import cv2
from collections import ChainMap
import numpy as np
import argparse

from comms import SerialLine
from hands3d import HandDet3D

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--loadfile', type=str, 
                    help="Config file to load", required=True)
parser.add_argument('-c', '--calfile', type=str, 
                    help="Config file to save to", required=False)
parser.add_argument('-i', '--iter', type=int, default=30, 
                    help="Iterations for calibration", required=False)
args = parser.parse_args()
calIter = args.iter
fing_count=6
iter = 0

def norm(v):
    if v>1: return 1
    if v<0: return 0
    return v

handDet3D = HandDet3D(args.loadfile)
calIs = False
serialLine = SerialLine(115200, fing_count-1, 0, 1)

cap = cv2.VideoCapture(2)#ext camera
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

        multiLandMarks = handDet3D.procImage(image)
        '''if multiLandMarks:
            for handWorldLandarks in multiLandMarks:
                plot.plot_landmarks2(
                    handWorldLandarks, handDet3D.mp_Hands.HAND_CONNECTIONS)'''
        if multiLandMarks:
            #for handLms in multiLandMarks:
            handLms = multiLandMarks[0]
            handDet3D.procHand(image, handLms)
            out = handDet3D.iterAngle()
            #print([round(ou, 4) for ou in out])
            out = [norm(v) for v in out]
            #print(out[3])
            out = [out[0], out[1], out[2], out[2], out[4], out[4]]#TEMP
            #serialLine.compVal(out)
            print(handDet3D.fingers[0].joints[0].p1.z)
            serialLine.sendVal(out)
            #print(out)

            if args.calfile:
                cal = handDet3D.calib(calIter)
                if cal and calIs==False:
                    calIs=True
                    #print(cal)
                    print([{key:[np.max(value2) for value2 in value]} for key,value in cal.items()])
                    handDet3D.saveConfig(
                        args.calfile, 
                        dict(ChainMap(*[{key:[np.min(value2) for value2 in value]} for key,value in cal.items()])), 
                        dict(ChainMap(*[{key:[np.max(value2) for value2 in value]} for key,value in cal.items()])), 
                    )
        iter+=1

        cv2.imshow("Counting number of fingers", image)
        cv2.waitKey(1)
        #print(1/(time.time()-t))
        avg_t = (avg_t*iter+( 1/(time.time()-t) ))/(iter+1)
        #print(f"{avg_t:.2f}, {(1/(time.time()-t)):.2f}")

except KeyboardInterrupt:
    cap.release()
