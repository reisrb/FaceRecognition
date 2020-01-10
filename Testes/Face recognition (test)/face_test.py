import numpy as np
import cv2
import os

subjects = ["", "Rafael Reis", "Elvis Presley"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)