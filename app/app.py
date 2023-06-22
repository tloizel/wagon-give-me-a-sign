import streamlit as st
import cv2
import mediapipe as mp
from model import load_model, predict_model_ml
import pandas as pd
from data_proc import preproc_predict
#from tensorflow.keras.models import load_model
import string
ALPHABET = list(string.ascii_lowercase)

st.markdown("""# Give me a sign
""")


def main():

    #load model
    model = load_model('models/dense_1')
    # Initialise MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7)

    # Initialise MediaPipe Drawing
    mp_draw = mp.solutions.drawing_utils

    # Ouvre la webcam
    cap = cv2.VideoCapture(0)
    # Set the width and height of the video capture
    cap.set(3, 640)
    cap.set(4, 480)

    # Create a Streamlit placeholder to display the video
    video_placeholder = st.empty()

    while True:

        ret, frame = cap.read()

        # Convertit l'image en RGB pour MediaPipe Hands
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Fait passer l'image à travers le modèle
        results = hands.process(rgb_image)

        # Dessine les résultats
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(rgb_image,
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

            #draw rectangle around hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 0), 4)

            #predict and show prediction
            coords_df = preproc_predict(rgb_image, {'mp_hands': mp_hands, 'hands': hands, 'results': results})
            if coords_df is None:
                pass
            else:
                pred = model.predict(coords_df)

                res = 'none' if pred is None else pred

                res = res[0].tolist()
                max_value = max(res)
                max_index = res.index(max_value)
                if max_value>0.80:
                    answer = f"The letter is {ALPHABET[max_index]} at {round(max_value,2)}%"
                else:
                    answer = "None"


                cv2.putText(rgb_image,
                            answer,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA)


        # Display the frame in Streamlit
        video_placeholder.image(rgb_image, channels="RGB")
        # # Affiche l'image
        # cv2.imshow('Hand Tracking', frame)

        # Sortir de la boucle si 'q' est appuyé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Libère la webcam et ferme les fenêtres d'affichage
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
