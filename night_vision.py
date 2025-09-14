# Night vision enhancement utils
import cv2

def apply_night_vision(frame):
    b, g, r = cv2.split(frame)
    g = cv2.equalizeHist(g)
    return cv2.merge((b, g, r))
