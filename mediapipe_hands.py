import cv2, time
import mediapipe as mp
import numpy as np
from threading import Thread
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# ==== TODO ====
# Camera IO polling with threading
# Flask microserver
# Reduce frame size before inputing into MediaPipe then
# draw on full size frame.

 #Initialize the Flask app
def image_resize(image, scale):
    # Resizing images for better mp performance
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


class CameraFetch:
  def __init__(self, src=0, width=800, height=400):
    self.capture = cv2.VideoCapture(src)
    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Start the thread to read frames from the video stream
    self.thread = Thread(target=self.update, args=())
    self.thread.daemon = True
    self.thread.start()
    self.frame = cv2.imread('loading-2.png', cv2.IMREAD_COLOR)
    self.resized = cv2.imread('loading-2.png', cv2.IMREAD_COLOR)

  def __del__(self):
    self.capture.release()
  # def img_resize(self, image, scale):
  #   # Resizing images for better mp performance
  #   width = int(image.shape[1] * scale / 100)
  #   height = int(image.shape[0] * scale / 100)
  #   dim = (width, height)
  #   self.resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  #   # print(self.resized.shape)

  def update(self):
    # Read the next frame from the stream in a different thread
    while True:
        if self.capture.isOpened():
            (self.status, self.read) = self.capture.read()
            self.frame = cv2.flip(self.read,1)
            self.resized = image_resize(self.frame, 50)
        
        time.sleep(.02) # 60 fps MAX
    
    
  def mp_draw(self):
    # Draw the hand annotations on the image.    
      self.frame.flags.writeable = True
      #self.frame = cv2.cvtColor( self.frame, cv2.COLOR_RGB2BGR)
      if self.results.multi_hand_landmarks:
        print(self.fingerCount(self.results.multi_hand_landmarks[0]))
        for hand_landmarks in self.results.multi_hand_landmarks:
            #print(hand_landmarks)
            #print(self.fingerCount(hand_landmarks))
            if self.results.multi_hand_landmarks is not None:
                mp_drawing.draw_landmarks(
                self.frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
    
  def mp_process(self):
   with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
      self.resized.flags.writeable = False
      # print(self.resized.shape)
      # self.resized = cv2.cvtColor(self.resized, cv2.COLOR_BGR2RGB)
      self.results = hands.process(self.resized)
  
  def fingerCount(self, handLandmark) -> list:
    # define fingers per hand
    # print(handLandmark)
    fingers = [mp_hands.HandLandmark.THUMB_TIP, 
              mp_hands.HandLandmark.INDEX_FINGER_TIP,
              mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
              mp_hands.HandLandmark.RING_FINGER_TIP,
              mp_hands.HandLandmark.PINKY_TIP]
    # empty array to polulat finger states
    finger_state = []
    # defined wrist and index mcp coordinates for distance measurement
    wrist = handLandmark.landmark[mp_hands.HandLandmark.WRIST]
    pinkyMcp = handLandmark.landmark[mp_hands.HandLandmark.PINKY_MCP]
    # iterare fingers
    for finger in fingers:
        # finger tip coordinates
        tip = handLandmark.landmark[finger]
        # finger second joint coordinates
        pip = handLandmark.landmark[finger -2]
        # if finger is not thumb measure distance to wrist coordinate
        if finger != mp_hands.HandLandmark.THUMB_TIP:
            tipDist = np.linalg.norm(np.subtract([tip.x,tip.y],[wrist.x,wrist.y]))
            pipDist = np.linalg.norm(np.subtract([pip.x,pip.y], [wrist.x,wrist.y]))
            # if thumb measure distance to index finger mcp
        else:
            tipDist = np.linalg.norm(np.subtract([tip.x,tip.y],[pinkyMcp.x,pinkyMcp.y]))
            pipDist = np.linalg.norm(np.subtract([pip.x,pip.y], [pinkyMcp.x,pinkyMcp.y]))
            # if finger tip is higher than finger 2nd joint then finger is open ( append 1 to list)
        if tipDist > pipDist:
            finger_state.append(1)
        else:
            finger_state.append(0)
    return finger_state

  def mp_imshow(self):
        cv2.imshow('MediaPipe Hands', cv2.flip(self.frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
  def mp_byte_array(self):
    self.mp_process()
    self.mp_draw()
    ret, buffer = cv2.imencode('.jpg', self.frame)
    # print(buffer.tobytes())
    return buffer.tobytes()
    






if __name__ == '__main__': 
    cameraCapture = CameraFetch(width=1920, height=1080)

    while True:
        try:
            cameraCapture.mp_process()
            cameraCapture.mp_draw()
            cameraCapture.mp_imshow()
        except AttributeError:
            pass