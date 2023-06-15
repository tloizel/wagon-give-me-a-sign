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

    # Affiche l'image
    cv2.imshow('Hand Tracking', frame)

    # Sortir de la boucle si 'q' est appuyé
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libère la webcam et ferme les fenêtres d'affichage
#cap.release()
#cv2.destroyAllWindows()
