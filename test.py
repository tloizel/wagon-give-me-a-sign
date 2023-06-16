import cv2
import mediapipe as mp


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


        for hand_landmarks in results.multi_hand_landmarks:
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



    # Affiche l'image
    cv2.imshow('Hand Tracking', frame)

    # Sortir de la boucle si 'q' est appuyé
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libère la webcam et ferme les fenêtres d'affichage
#cap.release()
#cv2.destroyAllWindows()
