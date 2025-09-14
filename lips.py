import cv2
import dlib
import numpy as np
import pygame
import os
import datetime
from drowsiness_detection.utils.night_vision import apply_night_vision

def run_lip_detection():
    detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join('resources', 'shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(predictor_path)

    lip_distance_threshold = 18
    frame_counter = 0
    drowsy = False

    pygame.mixer.init()
    pygame.mixer.music.load(os.path.join('resources', 'music.wav'))

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = datetime.datetime.now().time()
        if current_time < datetime.time(6, 0) or current_time > datetime.time(19, 0):
            frame = apply_night_vision(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            shape = predictor(gray, face)
            shape_np = np.zeros((68, 2), dtype=int)
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            lip_dist = lip_distance(shape_np)
            for x, y in shape_np:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            if lip_dist > lip_distance_threshold:
                frame_counter += 1
                if frame_counter > 10 and not drowsy:
                    pygame.mixer.music.play(-1)
                    drowsy = True
            else:
                frame_counter = 0
                if drowsy:
                    pygame.mixer.music.stop()
                    drowsy = False
        cv2.imshow("Lips Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    bottom_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    return abs(top_mean[1]-bottom_mean[1])
