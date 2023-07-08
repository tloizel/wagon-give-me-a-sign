from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, rectangle, addWeighted, putText, FONT_HERSHEY_PLAIN, LINE_AA
from data_proc import preproc_predict
from game import translate_words
from string import ascii_lowercase

ALPHABET = list(ascii_lowercase)

ALPHABET_EXTRA = ALPHABET
ALPHABET_EXTRA.extend(['fuck', 'love', 'space', 'back'])
ALPHABET_EXTRA.sort()


def process(image, mp_drawing, mp_drawing_styles, mp_hands, hands, model):

    # with lock:
    #     image = img_container["img"]

    image.flags.writeable = False
    image = cvtColor(image, COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True

    image = cvtColor(image, COLOR_RGB2BGR)

    answer = 'No letter'
    text = 'No letter'

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        #for rectangle later
        H, W, _ = image.shape
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
        rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)


        #predict and show prediction
        coords_df = preproc_predict(image, {'mp_hands': mp_hands, 'hands': hands, 'results': results})

        pred = model.predict_proba(coords_df)

        pred = pred[0].tolist()
        max_value = max(pred)
        max_index = pred.index(max_value)

        if max_value>0.50 and pred is not None:
            answer = translate_words(ALPHABET_EXTRA[max_index]).capitalize()
            text = f"{answer} ({round(max_value,2)*100}%)"
            # answer = f"{translate_words(pred[0]).capitalize()} ({round(max_value,2)*100}%)"


        # Draw the background rectangle with transparency
        overlay = image.copy()
        rectangle(image, (x1, y1-8), (x2, y1 - 34), (255, 255, 255), -1)
        alpha = 0.3
        image = addWeighted(overlay, alpha, image, 1 - alpha, 0)


        putText(image,
                    text,
                    (x1+5, y1 - 10),
                    FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 0),
                    2,
                    LINE_AA)

    return image, answer


def most_common(lst):
    return max(set(lst), key=lst.count)
