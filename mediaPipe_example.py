import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


#### TODO
# Camera IO polling with threading
# Flask microserver
# Reduce frame size before inputing into MediaPipe then
#draw on full size frame.
 


def fingerCount(handLandmark) -> list:

  # define fingers per hand
  fingers = [mp_hands.HandLandmark.THUMB_TIP, 
             mp_hands.HandLandmark.INDEX_FINGER_TIP,
             mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
             mp_hands.HandLandmark.RING_FINGER_TIP,
             mp_hands.HandLandmark.PINKY_TIP]
  # empty array to polulat finger states
  finger_state = []
  # defined wrist and index mcp coordinates for distance measurement
  wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
  wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
  indexMcpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
  indexMcptY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
  # iterare fingers
  for finger in fingers:
    # finger tip coordinates
    tipX = handLandmark.landmark[finger].x
    tipY = handLandmark.landmark[finger].y
    # finger second joint coordinates
    pipX = handLandmark.landmark[finger -2].x
    pipY = handLandmark.landmark[finger -2].y
    # if finger is not thumb measure distance to wrist coordinate
    if finger != mp_hands.HandLandmark.THUMB_TIP:
      tipDist = np.linalg.norm(np.subtract([tipX,tipY],[wristX,wristY]))
      pipDist = np.linalg.norm(np.subtract([pipX,pipY], [wristX,wristY]))
    # if thumb measure distance to index finger mcp
    else:
      tipDist = np.linalg.norm(np.subtract([tipX,tipY],[indexMcpX,indexMcptY]))
      pipDist = np.linalg.norm(np.subtract([pipX,pipY], [indexMcpX,indexMcptY]))
    # if finger tip is higher than finger 2nd joint then finger is open ( append 1 to list)
    if tipDist > pipDist:
      finger_state.append(1)
    else:
      finger_state.append(0)
  return finger_state



cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

        # Draw the hand annotations on the image.
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # if results.multi_hand_landmarks is not None:
    #  print(results.multi_hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        if results.multi_hand_landmarks is not None:
          print(fingerCount(hand_landmarks))
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
