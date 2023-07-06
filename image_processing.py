import cv2
from data_proc import preproc_predict
from game import translate_words


def process(image, mp_drawing, mp_drawing_styles, mp_hands, hands, model):

    # with lock:
    #     image = img_container["img"]

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    answer = 'No hand'

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
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)


        #predict and show prediction
        coords_df = preproc_predict(image, {'mp_hands': mp_hands, 'hands': hands, 'results': results})

        if coords_df is None:
            pass
        else:
            # pred = model.predict_proba(coords_df)
            pred = model.predict(coords_df)


            res = 'none' if pred is None else pred

            # res = res[0].tolist()
            # max_value = max(res)
            # max_index = res.index(max_value)

            # if max_value>0.90:
            if res is not None:
                # answer = f"{translate_words(res[0]).capitalize()} ({round(max_value,2)*100}%)"
                answer = translate_words(res[0]).capitalize()

            else:
                answer = "No letter"

        # Draw the background rectangle with white color
        overlay = image.copy()

        cv2.rectangle(image, (x1, y1-8), (x2, y1 - 34), (255, 255, 255), -1)

        alpha = 0.3

        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        cv2.putText(image,
                    answer,
                    (x1+5, y1 - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA)

    return image, answer


def most_common(lst):
    return max(set(lst), key=lst.count)
