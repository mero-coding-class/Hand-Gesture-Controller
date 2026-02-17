import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np

# =======================
# Initial Setup
# =======================
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =======================
# Control Parameters
# =======================
smoothening = 2
frame_margin = 100

prev_x, prev_y = 0, 0
dragging = False

last_right_click = 0
right_click_delay = 0.8

last_enter_time = 0
enter_delay = 1.2

# =======================
# Helper Functions
# =======================
def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def is_open_palm(lm):
    fingers = []

    # Thumb (x-axis)
    fingers.append(lm[4].x < lm[3].x)

    # Other fingers (y-axis)
    fingers.append(lm[8].y < lm[6].y)    # Index
    fingers.append(lm[12].y < lm[10].y)  # Middle
    fingers.append(lm[16].y < lm[14].y)  # Ring
    fingers.append(lm[20].y < lm[18].y)  # Pinky

    return all(fingers)

# =======================
# Main Loop
# =======================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # Index fingertip position
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        # Limit active area (speed boost)
        x = np.clip(x, frame_margin, w - frame_margin)
        y = np.clip(y, frame_margin, h - frame_margin)

        # Map to screen
        screen_x = np.interp(x, (frame_margin, w - frame_margin), (0, screen_w))
        screen_y = np.interp(y, (frame_margin, h - frame_margin), (0, screen_h))

        # Smooth movement
        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening
        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

        # Distances for gestures
        pinch_index = distance(lm[4], lm[8])
        pinch_middle = distance(lm[4], lm[12])

        current_time = time.time()

        # =======================
        # Left Click & Drag
        # =======================
        if pinch_index < 0.03:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

        # =======================
        # Right Click
        # =======================
        if pinch_middle < 0.03 and current_time - last_right_click > right_click_delay:
            pyautogui.rightClick()
            last_right_click = current_time

        # =======================
        # ENTER key (Open Palm)
        # =======================
        if is_open_palm(lm) and current_time - last_enter_time > enter_delay:
            pyautogui.press("enter")
            last_enter_time = current_time

            cv2.putText(
                img,
                "ENTER",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        # Draw landmarks
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        # Cursor dot
        cv2.circle(img, (x, y), 8, (0, 255, 0), cv2.FILLED)

    # Active movement area (visual guide)
    cv2.rectangle(
        img,
        (frame_margin, frame_margin),
        (w - frame_margin, h - frame_margin),
        (255, 0, 255),
        2
    )

    cv2.imshow("Gesture Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
