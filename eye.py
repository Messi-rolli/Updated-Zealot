import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from drowsiness_detection.utils.night_vision import apply_night_vision
from drowsiness_detection.utils.notifications import send_email, send_sms
import datetime
from pygame import mixer
import os

def run_eye_detection():
    mixer.init()
    music_path = os.path.join('resources', 'music.wav')
    mixer.music.load(music_path)

    thresh = 0.25
    frame_check = 20
    flag = 0
    numk = 0

    predictor_path = os.path.join('resources', 'shape_predictor_68_face_landmarks.dat')
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(predictor_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = datetime.datetime.now().time()
        if current_time < datetime.time(6, 0) or current_time > datetime.time(19, 0):
            frame = apply_night_vision(frame)
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            for eye in [leftEye, rightEye]:
                eyeHull = cv2.convexHull(eye)
                cv2.drawContours(frame, [eyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    numk += 1
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if numk == 11:
                        send_email("Drowsiness Alert",
                            "Drowsiness detected, please check the driver",
                            os.environ.get('ALERT_EMAIL'))
                        send_sms("Drowsiness detected", os.environ.get('TWILIO_FROM'),
                                 os.environ.get('TWILIO_TO'))
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                flag = 0
                if mixer.music.get_busy():
                    mixer.music.stop()

        cv2.imshow("Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)
