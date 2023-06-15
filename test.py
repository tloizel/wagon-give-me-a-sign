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
            cv2.putText(frame, "Absolute",
                        (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            cv2.putText(frame, f"X : {round(thumb_tip_x,2)}",
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            cv2.putText(frame, f"Y : {round(thumb_tip_y,2)}",
                        (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            cv2.putText(frame, f"Z : {round(thumb_tip_z,2)}",
                        (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            # Use the relative coordinates
            cv2.putText(frame, "Relative",
                        (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            cv2.putText(frame, f"X : {round(thumb_tip_x - wrist_x,2)}",
                        (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            cv2.putText(frame, f"Y : {round(thumb_tip_y - wrist_y,2)}",
                        (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

            cv2.putText(frame, f"Z : {round(thumb_tip_z - wrist_z,2)}",
                        (500, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)

    # Affiche l'image
    cv2.imshow('Hand Tracking', frame)

    # Sortir de la boucle si 'q' est appuyé
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libère la webcam et ferme les fenêtres d'affichage
#cap.release()
#cv2.destroyAllWindows()
