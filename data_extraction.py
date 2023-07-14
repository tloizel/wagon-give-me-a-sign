# import os
import mediapipe as mp
from cv2 import imread, cvtColor, COLOR_BGR2RGB
# import matplotlib.pyplot as plt
import pandas as pd
# import ipdb
# import numpy as np


def get_coordinates(image, processed_hand_dict=None):
    """
    Récupère toutes les coords de la main d'une image et stocke dans un dictionnaire
    possible de passer le path d'une image ou alors les résultats d'une main déjà processé, pour alléger le code
    Sinon problème de mémoire.
    dict : {'mp_hands': mp_hands, 'hands': hands, 'results': results}
    """
    if processed_hand_dict is not None:
        #from frame
        mp_hands = processed_hand_dict['mp_hands']
        hands = processed_hand_dict['hands']
        results = processed_hand_dict['results']
    else:
        #from path, for collection
        img = imread(image)
        img_rgb = cvtColor(img, COLOR_BGR2RGB)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.75)
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

                coords[f"x_{i}"]= -(x - wrist_x)
                coords[f"y_{i}"]= y - wrist_y
                coords[f"z_{i}"]= z - wrist_z

    return coords



def get_coordinates_from_collection():
    import os

    """
    Récupère toutes les coords des mains de raw_data et stocke dans un dataframe
    """
    DATA_DIR = './raw_data'

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7)

    data = []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            img = os.path.join(DATA_DIR, dir_, img_path)

            image = imread(img)
            rgb_image = cvtColor(image, COLOR_BGR2RGB)
            # Fait passer l'image à travers le modèle
            results = hands.process(rgb_image)

            coords = get_coordinates(img, {'mp_hands': mp_hands, 'hands': hands, 'results': results})
            coords['target']=dir_
            data.append(coords)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    get_coordinates_from_collection()
