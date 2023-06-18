import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def get_coordinates(image):
    """
    Récupère toutes les coords de la main d'une image et stocke dans un dictionnaire
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    img = cv2.imread(image)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    coords={}

    if results.multi_hand_landmarks:
        # total_landmarks = sum(len(hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)
        # print(total_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z


            for i in range(len(hand_landmarks.landmark)):
                # same order for everyone
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                coords[f"x_{i}"]= x - wrist_x
                coords[f"y_{i}"]= y - wrist_y
                coords[f"z_{i}"]= z - wrist_z

    return coords



def get_coordinates_from_collection():
    """
    Récupère toutes les coords des mains de raw_data et stocke dans un dataframe
    """
    DATA_DIR = './raw_data'

    data = []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            img = os.path.join(DATA_DIR, dir_, img_path)
            coords = get_coordinates(img)
            coords['target']=dir_
            data.append(coords)

    df = pd.DataFrame(data)
    print(df)

if __name__ == "__main__":
    get_coordinates_from_collection()
