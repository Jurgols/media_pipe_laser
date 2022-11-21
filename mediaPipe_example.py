import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



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
  wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
  pinkyMcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
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
