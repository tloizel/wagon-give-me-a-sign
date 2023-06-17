import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

import csv
import pandas as pd


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './raw_data'

data = []
labels = []

#toto = os.listdir(DATA_DIR)
#print(toto)

data_relative = []
#x_ = []
#y_ = []
#z_ = []

for dir_ in os.listdir(DATA_DIR):
    print(f'on est dans le folder {dir_}')

    if dir_ not in ['.DS_Store']:  # Dossier de log qui provoque une erreur

        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

            if img_path not in ['.DS_Store']: # Dossier de log qui provoque une erreur

                #data_relative = []

                #x_ = []
                #y_ = []
                #z_ = []


                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                print(f'image : {img_path} dans le folder {dir_}')

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            z = hand_landmarks.landmark[i].z
                            #viz = hand_landmarks.landmark[i].visibility

                            #x_.append(x)
                            #y_.append(y)
                            #z_.append(z)

                        #for i in range(len(hand_landmarks.landmark)):
                            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                            wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                            data_relative.append(str(x - wrist_x))
                            data_relative.append(str(y - wrist_y))
                            data_relative.append(str(z - wrist_z))
                            #data_relative.append(str(dir_))   # ajout de la classe

                print(f'on va append : {data_relative} de longueur : {len(data_relative)}')

                data_relative.append(dir_)
                data.append(data_relative)
                labels.append(dir_)
                data_relative = []

        #data_relative = []
        #x_ = []
        #y_ = []
        #z_ = []

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False, sep=';')

print(f'longueur par ligne : {len(data[0])}, la data : {data[0]}')
