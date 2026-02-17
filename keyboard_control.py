import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os
import time

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(1)

# Cooldown tracker
last_gesture = None
last_action_time = 0
cooldown_time = 0.8  # seconds

# OPTIONAL: file to open directly
FILE_PATH = r"C:\Users\Public\Documents"  # change if needed

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    gesture = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))

            wrist = landmarks[0]
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # ---------------- GESTURES ---------------- #

            # LEFT / RIGHT
            if index_tip[0] < w // 3:
                gesture = "LEFT"
            elif index_tip[0] > 2 * w // 3:
                gesture = "RIGHT"

            # UP (jump)
            elif wrist[1] - index_tip[1] > 120:
                gesture = "UP"

            # DOWN (fist)
            elif abs(index_tip[1] - wrist[1]) < 40 and abs(index_tip[0] - thumb_tip[0]) < 40:
                gesture = "DOWN"

            # PINCH (Open file / Enter)
            elif distance(index_tip, thumb_tip) < 35:
                gesture = "PINCH"

            # ---------------- ACTION ---------------- #
            current_time = time.time()

            if gesture and (gesture != last_gesture or (current_time - last_action_time) > cooldown_time):

                if gesture == "LEFT":
                    pyautogui.press("left")
                    # time.sleep(1)

                elif gesture == "RIGHT":
                    pyautogui.press("right")
                    # time.sleep(1)

                elif gesture == "UP":
                    pyautogui.press("up")
                    # time.sleep(1)

                elif gesture == "DOWN":
                    pyautogui.press("down")
                    # time.sleep(1)

                elif gesture == "PINCH":
                    # OPTION 1: Press Enter
                    pyautogui.press("enter")
                    # time.sleep(1)

                    # OPTION 2: Open file directly (uncomment if needed)
                    # os.startfile(FILE_PATH)

                last_gesture = gesture
                last_action_time = current_time

            # Display gesture
            if gesture:
                cv2.putText(img, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
