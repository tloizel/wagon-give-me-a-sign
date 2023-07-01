import streamlit as st
import cv2
import mediapipe as mp
import time
from model import load_model_ml, predict_model_ml
# import pandas as pd
# from data_proc import preproc_predict
#from tensorflow.lite.keras.models import load_model
import string
ALPHABET = list(string.ascii_lowercase)
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from game import random_letter
import ipdb
from data_extraction import get_coordinates
import pandas as pd

def preproc_predict(image, processed_hand_dict):
    """
    Get coords
    Remove the first three columns x_0, y_0, z_0
    Return coods
    """
    coords = get_coordinates(image, processed_hand_dict)
    if not coords:
        return None
    else :
        coords = pd.DataFrame(coords, index=[0])
        coords = coords.drop(['x_0','y_0','z_0'], axis=1)
        return coords

import threading

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame


model = load_model_ml()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7)
num_consecutive_frames = 10

def process_out_of_class(image, model):

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


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

            predicting_letter = res[0].capitalize()
            return predicting_letter

class VideoProcessor:

    def process(image):

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


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

                predicting_letter = res[0].capitalize()

                # res = res[0].tolist()
                # max_value = max(res)
                # max_index = res.index(max_value)

                # if max_value>0.90:
                if res is not None:
                    # answer = f"{ALPHABET[max_index].capitalize()} ({round(max_value,2)*100}%)"
                    answer = res[0]

                else:
                    answer = "No letter"


                cv2.putText(image,
                            answer,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA)


        # Make a prediction for the current frame

        return image

    def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = self.process(img)
            with lock:
                img_container["img"] = img
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    goal = random_letter()
    st.markdown(f"# Do a {goal}")

    RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7)
    # load model

    # create instance of video stream with processing

    # Streamlit UI
    st.title("Letter Prediction")


    # Stream
    ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        # video_frame_callback=video_frame_callback,
        )

    predictions_list = []
    counter = 0
    result = st.empty()
    while ctx.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue
        pred = process_out_of_class(img, model)
        if pred is None:
            continue
        predictions_list.append(pred)
        counter += 1
        if counter == 50:
            letter = most_common(predictions_list)
            if letter == goal :
                goal = random_letter()
                result = st.write(f"Congratulations, now do {goal}")
                counter = 0
                predictions_list = []



def most_common(lst):
    return max(set(lst), key=lst.count)

if __name__ == "__main__":
    main()
