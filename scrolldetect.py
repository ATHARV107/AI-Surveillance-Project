import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Cooldown tracker
last_action_time = 0
cooldown = 0  # seconds

# Helper to count fingers up
def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(handLms)

            current_time = time.time()

            if current_time - last_action_time > cooldown:
                # Gesture actions
                if fingers_up == 0:
                    print("Pause/Play Video")
                    pyautogui.press('space')  # Pause/Play
                    last_action_time = current_time

                elif fingers_up == 5:
                    print("Scroll Up")
                    pyautogui.scroll(50)
                    last_action_time = current_time

                elif fingers_up == 2:
                    print("Scroll Down")
                    pyautogui.scroll(-50)
                    last_action_time = current_time

                elif fingers_up == 1:
                    print("Volume Up")
                    pyautogui.press('volumeup')
                    last_action_time = current_time

                elif fingers_up == 3:
                    print("Volume Down")
                    pyautogui.press('volumedown')
                    last_action_time = current_time

    cv2.imshow("YouTube Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
