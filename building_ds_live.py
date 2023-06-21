import os
import cv2
import string
import uuid
import pandas as pd
import mediapipe as mp

def get_coordinates_live(frame):
    """
    Récupère toutes les coords de la main d'une image et stocke dans un dictionnaire
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    #img = cv2.imread(image)

    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    return coords





DATA_DIR = './raw_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

alphabet = list(string.ascii_lowercase)
alphabet = ['a', 'b', 'c']
dataset_size = 10 #each of us should do 100, 400 in total per letter


data = []
labels = []
cap = cv2.VideoCapture(0)
for j in alphabet:
    # if not os.path.exists(os.path.join(DATA_DIR, str(j))):
    #     os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "{j}" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord(j):
            break

    counter = 0
    while counter < dataset_size:
        coords = {}
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(40)
        #cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(str(uuid.uuid1()))), frame)
        coords = get_coordinates_live(frame)

        #print(f'coords est de type : {type(coords)} -------------')


        if coords:
            coords['target'] = j
            data.append(coords)
            labels.append(j)
            counter += 1
            print(f'COORD no {counter} pour la target {j} avec les coord : {coords}')
        else:
            print(f' ----------- data non collectée, coord vide : {coords}')

cap.release()
cv2.destroyAllWindows()

coord_live = pd.DataFrame(data)
totot = pd.DataFrame(labels)

print('---------------------------')
print('---------------------------')

print(coord_live.head(3))

print('---------------------------')
print('---------------------------')

print(totot.head(3))

coord_live.to_csv("csv/coord_live.csv", index=False)
