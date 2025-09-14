import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
from ultralytics import YOLO
import threading
import queue
import time

# Thresholds (tune as needed)
THUMB_INDEX_THRESHOLD = 30       # pixels, for phone grip detection
EYE_AR_THRESHOLD = 0.23          # EAR threshold for drowsiness
EYE_AR_CONSEC_FRAMES = 15        # consecutive frames threshold for drowsiness alarm
MOUTH_AR_THRESHOLD = 0.7         # mouth aspect ratio threshold for yawn detection

def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

def eye_aspect_ratio(landmarks, w, h):
    # Eye landmarks (MediaPipe FaceMesh)
    # Using 6 points for each eye from typical indexes (adjust if needed)
    def ar(pts):
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h1 = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h1)
    left_idxs = [33, 133, 159, 145, 153, 154]
    right_idxs = [263, 362, 386, 374, 380, 381]
    left_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_idxs])
    right_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_idxs])
    left_ar = ar(left_pts)
    right_ar = ar(right_pts)
    ear = (left_ar + right_ar) / 2.0
    return ear

def mouth_aspect_ratio(landmarks, w, h):
    upper = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lower = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left = np.array([landmarks[78].x * w, landmarks[78].y * h])
    right = np.array([landmarks[308].x * w, landmarks[308].y * h])
    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal

def detect_phone_grip(hand_landmarks, w, h):
    thumb_tip = np.array([hand_landmarks[4].x * w, hand_landmarks[4].y * h])
    index_mcp = np.array([hand_landmarks[7].x * w, hand_landmarks[7].y * h])
    dist = np.linalg.norm(thumb_tip - index_mcp)
    return dist < THUMB_INDEX_THRESHOLD, dist

def run_multi_detection():
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Load YOLOv8 phone detection model
    yolo_model = YOLO("yolov8n.pt")

    # Initialize pygame sounds
    pygame.mixer.init()
    resources_dir = 'resources'
    phone_sound_path = os.path.join(resources_dir, 'phone.wav')
    drowsy_sound_path = os.path.join(resources_dir, 'drowsy.wav')
    yawn_sound_path = os.path.join(resources_dir, 'yawn.wav')

    phone_sound = drowsy_sound = yawn_sound = None
    try:
        phone_sound = pygame.mixer.Sound(phone_sound_path)
    except pygame.error as e:
        print(f"Could not load phone sound: {e}")
    try:
        drowsy_sound = pygame.mixer.Sound(drowsy_sound_path)
    except pygame.error as e:
        print(f"Could not load drowsy sound: {e}")
    try:
        yawn_sound = pygame.mixer.Sound(yawn_sound_path)
    except pygame.error as e:
        print(f"Could not load yawn sound: {e}")

    phone_alarm_on = False
    drowsy_alarm_on = False
    yawn_alarm_on = False

    frame_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    cap = cv2.VideoCapture(0)
    reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event))
    reader_thread.start()

    drowsy_counter = 0

    try:
        while not stop_event.is_set():
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = frame_queue.get()
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame_rgb)
            yolo_results = yolo_model(frame)

            # Phone detection
            phone_boxes = []
            for r in yolo_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 67 and conf > 0.5:
                        coords = box.xyxy[0].cpu().numpy()
                        phone_boxes.append(coords)

            # Hand grip detection
            hand_holding_pose = False
            if results.left_hand_landmarks:
                grip, dist_left = detect_phone_grip(results.left_hand_landmarks.landmark, w, h)
                cv2.putText(frame, f"L Grip Dist: {dist_left:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
                if grip:
                    hand_holding_pose = True
            if results.right_hand_landmarks:
                grip, dist_right = detect_phone_grip(results.right_hand_landmarks.landmark, w, h)
                cv2.putText(frame, f"R Grip Dist: {dist_right:.1f}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
                if grip:
                    hand_holding_pose = True

            phone_alert = hand_holding_pose and len(phone_boxes) > 0
            if phone_alert:
                cv2.putText(frame, "Phone Usage Detected!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if phone_sound and not phone_alarm_on:
                    phone_sound.play(loops=-1)
                    phone_alarm_on = True
            else:
                if phone_sound and phone_alarm_on:
                    phone_sound.stop()
                    phone_alarm_on = False

            for box in phone_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'Phone', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Drowsiness detection using EAR
            drowsy_alert = False
            if results.face_landmarks:
                ear = eye_aspect_ratio(results.face_landmarks.landmark, w, h)
                cv2.putText(frame, f"EAR: {ear:.2f}", (w - 160, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                if ear < EYE_AR_THRESHOLD:
                    drowsy_counter += 1
                    if drowsy_counter >= EYE_AR_CONSEC_FRAMES:
                        drowsy_alert = True
                else:
                    drowsy_counter = 0

                if drowsy_alert:
                    cv2.putText(frame, "Drowsiness Detected!", (w - 260, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if drowsy_sound and not drowsy_alarm_on:
                        drowsy_sound.play(loops=-1)
                        drowsy_alarm_on = True
                else:
                    if drowsy_sound and drowsy_alarm_on:
                        drowsy_sound.stop()
                        drowsy_alarm_on = False

            # Yawn detection using MAR
            yawn_alert = False
            if results.face_landmarks:
                mar = mouth_aspect_ratio(results.face_landmarks.landmark, w, h)
                cv2.putText(frame, f"MAR: {mar:.2f}", (w - 160, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 60), 2)
                if mar > MOUTH_AR_THRESHOLD:
                    yawn_alert = True
                if yawn_alert:
                    cv2.putText(frame, "Yawn Detected!", (w - 220, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if yawn_sound and not yawn_alarm_on:
                        yawn_sound.play(loops=-1)
                        yawn_alarm_on = True
                else:
                    if yawn_sound and yawn_alarm_on:
                        yawn_sound.stop()
                        yawn_alarm_on = False

            # Draw landmarks
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

            cv2.imshow("Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    finally:
        stop_event.set()
        reader_thread.join()
        holistic.close()
        pygame.mixer.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_multi_detection()
