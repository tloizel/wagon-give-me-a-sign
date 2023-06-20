import cv2
import mediapipe as mp
#from model import load_model, predict_model_ml
import pandas as pd
from data_proc import preproc_predict
import ipdb
from tensorflow.keras.models import load_model
import string
ALPHABET = list(string.ascii_lowercase)

model = load_model('models/lstm_1')

#load model

# Initialise MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7)

# Initialise MediaPipe Drawing
mp_draw = mp.solutions.drawing_utils

# Ouvre la webcam
cap = cv2.VideoCapture(0)


while cap.isOpened():

    ret, frame = cap.read()

    # Convertit l'image en RGB pour MediaPipe Hands
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fait passer l'image à travers le modèle
    results = hands.process(rgb_image)

    # Dessine les résultats
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,
                                   hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                   mp.solutions.drawing_styles.get_default_hand_connections_style())

        #for rectangle later
        H, W, _ = frame.shape
        x_ = []
        y_ = []
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)


############################### FOR THUMB TEST DATA ############################################

        # Get the coordinates of the thumb_tip landmark
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

        # Get the coordinates of the wrist landmark
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z


        # Use the absolute coordinates
        c_a = 50
        for i in ["Absolute", ["X", thumb_tip_x],["Y", thumb_tip_y], ["Z", thumb_tip_z]]:

            if isinstance(i,str):
                print_ = i
            else:
                print_ =  f"{i[0]} : {round(i[1],2)}"

            cv2.putText(frame,
                        print_,
                        (100, c_a),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)
            c_a+=50

        # Use the relative coordinates
        c_r = 50
        for i in ["Relative", ["X", thumb_tip_x - wrist_x],["Y", thumb_tip_y - wrist_y], ["Z", thumb_tip_z - wrist_z]]:

            if isinstance(i,str):
                print_ = i
            else:
                print_ =  f"{i[0]}: {round(i[1],2)}"

            cv2.putText(frame,
                        print_,
                        (500, c_r),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)
            c_r+=50
###########################################################################



        #draw rectangle around hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        #predict and show prediction
        coords_df = preproc_predict(rgb_image, {'mp_hands': mp_hands, 'hands': hands, 'results': results})
        if coords_df is None:
            pass
        else:

            pred = model.predict(coords_df)

            res = 'none' if pred is None else pred
            print(res)

            res = res[0].tolist()
            max_value = max(res)
            max_index = res.index(max_value)
            if max_value>0.90:
                answer = f"The letter is {ALPHABET[max_index]} at {round(max_value,2)}%"
            else:
                answer = "None"



            cv2.putText(frame,
                        answer,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA)



    # Affiche l'image
    cv2.imshow('Hand Tracking', frame)

    # Sortir de la boucle si 'q' est appuyé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Libère la webcam et ferme les fenêtres d'affichage
cap.release()
cv2.destroyAllWindows()
