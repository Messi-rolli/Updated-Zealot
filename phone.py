import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
from ultralytics import YOLO
import threading
import queue
import time

def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # discard oldest
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

def run_phone_detection():
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Load YOLOv8 model
    yolo_model = YOLO("yolov8n.pt")

    # Initialize pygame and load alarm sound
    pygame.mixer.init()
    music_path = os.path.join('resources', 'music.wav')
    try:
        alarm_sound = pygame.mixer.Sound(music_path)
    except pygame.error as e:
        print(f"Error loading sound: {e}")
        return

    alarm_on = False

    frame_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    cap = cv2.VideoCapture(0)
    reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event))
    reader_thread.start()

    THUMB_INDEX_THRESHOLD = 30  # Pixels; adjust as necessary based on camera

    try:
        while not stop_event.is_set():
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame_rgb)

            yolo_results = yolo_model(frame)
            phone_boxes = []
            for r in yolo_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 67 and conf > 0.5:
                        coords = box.xyxy[0].cpu().numpy()
                        phone_boxes.append(coords)

            # Check hand "grip" for both hands
            h, w, _ = frame.shape
            hand_holding_pose = False
            thumb_index_dist_left = None
            thumb_index_dist_right = None

            if results.left_hand_landmarks:
                left_landmarks = results.left_hand_landmarks.landmark
                thumb_tip_left = np.array([left_landmarks[4].x * w, left_landmarks[4].y * h])
                index_mcp_left = np.array([left_landmarks[7].x * w, left_landmarks[7].y * h])
                thumb_index_dist_left = np.linalg.norm(thumb_tip_left - index_mcp_left)
                print(f"Left thumb-index: {thumb_index_dist_left:.2f}")
                if thumb_index_dist_left < THUMB_INDEX_THRESHOLD:
                    hand_holding_pose = True

                cv2.putText(frame, f"L Thumb-Idx Dist: {thumb_index_dist_left:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            if results.right_hand_landmarks:
                right_landmarks = results.right_hand_landmarks.landmark
                thumb_tip_right = np.array([right_landmarks[4].x * w, right_landmarks[4].y * h])
                index_mcp_right = np.array([right_landmarks[7].x * w, right_landmarks[7].y * h])
                thumb_index_dist_right = np.linalg.norm(thumb_tip_right - index_mcp_right)
                print(f"Right thumb-index: {thumb_index_dist_right:.2f}")
                if thumb_index_dist_right < THUMB_INDEX_THRESHOLD:
                    hand_holding_pose = True

                cv2.putText(frame, f"R Thumb-Idx Dist: {thumb_index_dist_right:.1f}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

            print(f"Phone boxes detected: {len(phone_boxes)}; Hand grip detected: {hand_holding_pose}")

            # Draw hands and phones
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            for box in phone_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'Phone', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            alarm_condition = hand_holding_pose and bool(phone_boxes)
            if alarm_condition:
                print("Alarm condition met (phone + hand grip) - playing sound")
                cv2.putText(frame, "Phone Usage Detected!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_sound.play(loops=-1)
                    alarm_on = True
            else:
                if alarm_on:
                    print("Alarm condition cleared - stopping sound")
                    alarm_sound.stop()
                    alarm_on = False

            cv2.imshow("Phone Detection", frame)
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
    run_phone_detection()
