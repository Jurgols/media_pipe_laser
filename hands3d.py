from locale import normalize
from symbol import xor_expr
from copy import copy
import numpy as np
import mediapipe as mp
import cv2
import typing
import json

min_detection_confidence=0.6
min_tracking_confidence=0.93

Point = typing.NewType("Point", object)
noNameC = 1

def avg(l):
    return sum(l)/len(l)
def normAngl(angl):
    if angl>np.pi:
        return np.pi
    return angl

class Point(object):
    def __init__(self, handDet3D, idx) -> None:
        self.handDet3D = handDet3D
        self.idx = idx
        self.x:float=0
        self.y:float=0
        self.z:float=0
    def __sub__(self, other:Point)->float:
        self.x = self.handDet3D.handList[self.idx][0]
        self.y = self.handDet3D.handList[self.idx][1]
        self.z = self.handDet3D.handList[self.idx][2]
        return np.sqrt( np.square(self.x-other.x) + np.square(self.y-other.y) + np.square(self.z-other.z) )

class Joint:
    def __init__(self, handDet3D, jObj) -> None:
        self.weight = float(jObj["weight"])
        self.p1:Point = Point(handDet3D, jObj["p1"])#B
        self.p2:Point = Point(handDet3D, jObj["p2"])#A
        self.p3:Point = Point(handDet3D, jObj["p3"])#C
        self.min = float(jObj["min"])
        self.max = float(jObj["max"])
        self.range_k = 1/(self.max-self.min)
        self.range_diff = self.min*self.range_k
    def saveConfig(self, min, max):
        return { "weight":str(self.weight), "p1":self.p1.idx, "p2":self.p2.idx, "p3":self.p3.idx, "min":str(min), "max":str(max) }
    def angle(self):
        a= self.p1 - self.p3
        b= self.p2 - self.p3
        c= self.p1 - self.p2

        angl = np.arccos((np.square(b)+np.square(c)-np.square(a))/(2*b*c))
        angl = np.abs(angl/np.pi)
        return 2*np.pi-angl if angl>np.pi else angl
    def adjAngle(self):
        return (self.angle()*self.range_k-self.range_diff)*self.weight
    def calib(self):
        try:
            self.tVal.append(self.angle())
        except AttributeError:
            self.tVal = []

class Finger:
    def __init__(self, handDet3D, jObjs) -> None:
        try:
            self.name = jObjs["name"]
        except KeyError:
            self.name = "noName"+str(noNameC)
            noNameC+=1
        self.joints: list[Joint] = [Joint(handDet3D, joint) for joint in jObjs["joints"]]
    def saveConfig(self, mins:dict, maxs:dict):
        return { 
            "name":self.name,
            "joints":[ self.joints[i].saveConfig(mins[self.name][i], maxs[self.name][i]) for i in range(len(self.joints)) ]
        }
    def angles(self):
        return sum([joint.adjAngle() for joint in self.joints])
    def calib(self):
        for v in self.joints:
            v.calib()
    def retCalib(self):
        return {self.name : [v.tVal for v in self.joints]}

class HandDet3D:
    def __init__(self, filename:str="") -> None:
        self.handList = [(0,0,0)]*21
        self.mp_Hands = mp.solutions.hands
        self.hands = self.mp_Hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.fingers: list[Finger] = []
        if filename != "":
            fObj = open(filename)
            self.loadConfig(json.load(fObj))#json.decoder.JSONDecodeError
            fObj.close()
        #self.joints: list[Joint] = [Joint(Point(self, 7), Point(self, 6), Point(self, 5), 0.4245842826489216, 0.8082585210151226)]
    def loadConfig(self, jObjs):
        for finger in jObjs["fingers"]:
            self.fingers.append(Finger(self, finger))
    def saveConfig(self, filename:str, mins:dict, maxs:dict):
        jOut = { "fingers" : [finger.saveConfig(mins, maxs) for finger in self.fingers]}
        print(jOut)
        f = open(filename, "w")
        json.dump(jOut, f)
        f.close()
    def procImage(self, image):
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(RGB_image)
        multiLandMarks = results.multi_hand_world_landmarks
        return multiLandMarks
    def procHand(self, image, handLms):
        for idx, lm in enumerate(handLms.landmark):
            self.handList[idx] = (lm.x, lm.y, lm.z)
        
    def iterAngle(self):#TEMP
        return [finger.angles() for finger in self.fingers]
    def calib(self, calibItrs=100, filename:str=None):
        try:
            if self.calibIt<self.calibLim:
                for v in self.fingers:
                    v.calib()
                self.calibIt += 1
            elif self.calibIt==self.calibLim:
                valDict = {}
                for v in self.fingers:
                    valDict = {**valDict, **v.retCalib()}
                return valDict
                #print(min(v.tVal), max(v.tVal))#0.4245842826489216 0.8082585210151226
        except AttributeError:
            self.calibIt=0
            self.calibLim=calibItrs