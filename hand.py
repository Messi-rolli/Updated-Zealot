import cv2
import mediapipe as mp
import pygame
import os

def run_hand_detection():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, 
                           min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    pygame.mixer.init()
    music_path = os.path.join('resources', 'music.wav')
    pygame.mixer.music.load(music_path)
    alarm_on = False

    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        hands_detected = results.multi_hand_landmarks is not None

        # Drawing landmarks and status text
        if hands_detected:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, "Hands on Steering", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Hands detected -> Stop the alarm if it is ON
            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False
        else:
            cv2.putText(image, "Hands Away! Alarm ON", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Hands NOT detected -> Start the alarm if not already ON
            if not alarm_on:
                pygame.mixer.music.play(-1)  # Play alarm continuously
                alarm_on = True

        cv2.imshow("Hand Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_detection()
