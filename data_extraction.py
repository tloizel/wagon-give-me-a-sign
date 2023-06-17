import os

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pandas as pd


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './raw_data'

data = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        coords={}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z

                    coords[f"x_{i}"]= x - wrist_x
                    coords[f"y_{i}"]= y - wrist_y
                    coords[f"z_{i}"]= z - wrist_z

            coords["target"]=dir_
            data.append(coords)


df = pd.DataFrame(data)
