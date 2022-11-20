import mediapipe as mp
from math import sqrt
import cv2

fC = 6

class HandDet2D:
    def __init__(self, finger_Coord, test_fing, thumb_Coord):
        self.finger_Coord = finger_Coord
        self.test_fing = test_fing# pedejais merogam
        self.thumb_Coord = thumb_Coord

        self.handList = []
        #raditajs, videjais, gredzen, maizais, ikskis, iksis rot
        self.high = [0.4488277490266238, 0.5079984394558149, 0.4786469190868624, 0.39748050088396736, 0.6136544334010876, 0.4806302582086738]
        self.low = [0.17978192544282778, 0.2667852642561041, 0.28300549114387213, 0.16496585243036269, 0.4017962598840015, 0.61888789338818]
        self.range_k = [1/(self.high[i]-self.low[i]) for i in range(fC)]
        self.range_diff = [self.low[i]*self.range_k[i] for i in range(fC)]
        self.avg = []
        self.calib = False

        self.mp_Hands = mp.solutions.hands
        self.hands = self.mp_Hands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    
    def procImage(self, image):
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(RGB_image)
        multiLandMarks = results.multi_hand_landmarks
        return multiLandMarks
    def procHand(self, image, handLms):
        self.mpDraw.draw_landmarks(image, handLms, self.mp_Hands.HAND_CONNECTIONS)
        for idx, lm in enumerate(handLms.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.handList.append((cx, cy))
            #for point in handList:
            #    cv2.circle(image, point, 10, (255, 255, 0), cv2.FILLED)
            #print(handList)
    def dist(self, f1):
        return sqrt((self.handList[f1[0]][0]-self.handList[f1[1]][0])**2+(self.handList[f1[1]][1]-self.handList[f1[0]][1])**2)
    def upDown(self):
            out_fing = [0, 0, 0, 0, 0, 0]
            upCount = 0
            for coordinate in self.finger_Coord:
                up = 0
                if self.handList[coordinate[0]][1] < self.handList[coordinate[1]][1]:
                    upCount += 1
                    up = 1
                indx = self.finger_Coord.index(coordinate)
                out_fing[indx] = up
            thumb_up = 0
            if self.handList[self.thumb_Coord[0]][0] > self.handList[self.thumb_Coord[1]][0]:
                upCount += 1
                thumb_up = 1
            out_fing[4] = thumb_up
            out_fing[5] = thumb_up
            return out_fing
    def iterFing(self):
        tt = []
        for coordinate in self.test_fing:
            tt.append(self.dist(coordinate))
        tt2 = [(t/tt[-1]) for t in tt[:-1]]
        val = [(tt2[i]*self.range_k[i]-self.range_diff[i]) for i in range(fC)]
        return val

    def calibFing(self):
        tt = [self.dist(t) for t in self.test_fing]
        tt2 = [(t/tt[-1]) for t in tt[:-1]]
        if not self.calib:
            self.avg = tt2
        self.avg = [(self.avg[i]*iter+tt2[i])/(iter+1) for i in range(len(self.avg))]
        return self.avg