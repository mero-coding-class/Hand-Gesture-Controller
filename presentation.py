import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

cap = cv2.VideoCapture(1)

cooldown = 1.5  # seconds between slide changes
last_action_time = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        current_time = time.time()

        # Get handedness classification
        hand_label = result.multi_handedness[0].classification[0].label

        # Only trigger if cooldown passed
        if (current_time - last_action_time) > cooldown:

            if hand_label == "Right":
                pyautogui.press("right")
                print("Right Hand → Next Slide")
                last_action_time = current_time

            elif hand_label == "Left":
                pyautogui.press("left")
                print("Left Hand → Previous Slide")
                last_action_time = current_time

    cv2.imshow("Hand Slide Control (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
