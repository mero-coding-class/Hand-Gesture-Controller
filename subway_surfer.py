import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Cooldown tracker
last_gesture = None
last_action_time = 0
cooldown_time = 0.8  # seconds (tweak this for sensitivity)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror for natural movement
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    gesture = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for id, lm in enumerate(handLms.landmark):
                landmarks.append((int(lm.x * w), int(lm.y * h)))

            # Key points
            wrist = landmarks[0]
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Compute relative positions
            # Left/Right movement based on x-position
            if index_tip[0] < w // 3:
                gesture = "LEFT"
            elif index_tip[0] > 2 * w // 3:
                gesture = "RIGHT"

            # Jump (hand raised)
            elif wrist[1] - index_tip[1] > 120:
                gesture = "JUMP"

            # Roll (fist detection)
            elif abs(index_tip[1] - wrist[1]) < 40 and abs(index_tip[0] - thumb_tip[0]) < 40:
                gesture = "ROLL"

            # Check cooldown before triggering
            current_time = time.time()
            if gesture and (gesture != last_gesture or (current_time - last_action_time) > cooldown_time):
                if gesture == "LEFT":
                    pyautogui.press('left')
                elif gesture == "RIGHT":
                    pyautogui.press('right')
                elif gesture == "JUMP":
                    pyautogui.press('up')
                elif gesture == "ROLL":
                    pyautogui.press('down')

                last_gesture = gesture
                last_action_time = current_time

            # Show gesture label on screen
            if gesture:
                cv2.putText(img, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display the camera feed
    cv2.imshow("Subway Surfers Gesture Control", img)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
