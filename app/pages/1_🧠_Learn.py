import streamlit as st
import cv2
import mediapipe as mp
# from model import load_model_ml, predict_model_ml
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import time

import sys
sys.path.append("./")  # Add the root directory to the Python path
from registry import load_model
from data_proc import preproc_predict
from game import random_letter, translate_words
from twilio_server import get_ice_servers

from image_processing import process, most_common


lock = threading.Lock()
img_container = {"img": None}


@st.cache_resource()
def define_hands():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7)
    return mp_drawing, mp_drawing_styles, mp_hands, hands

mp_drawing, mp_drawing_styles, mp_hands, hands = define_hands()


@st.cache_data()
def patience_while_i_load_the_model():
    # Load and return the model
    # return load_model(ml=True, model_name='random_forest_1')
    return load_model(ml=True, model_name='model_base_testing')

# model = load_model(ml=True, model_name='random_forest_1')
# model = load_model_ml()
model = patience_while_i_load_the_model()


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    img = process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[0]
    stream = av.VideoFrame.from_ndarray(img, format="bgr24")
    return stream



def main():

    # RTC_CONFIGURATION = RTCConfiguration(
    # {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    # )

    # Use this one for prod
    RTC_CONFIGURATION = {
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    }

    #Variables
    goal = 'A'
    predictions_list = []
    counter = 0

    # Streamlit UI

    col1, col2= st.columns([4,2])

    col1.title("Learn 🧠")
    col1.write("Time to get familiar with the alphabet")
    col1.write("")
    col1.write("")
    goal_text = col1.empty()
    goal_text.write(f"Show us the letter **{goal}**   👉")

    col2.write("")
    hint_image = col2.empty()
    image_path = f"https://raw.githubusercontent.com/tloizel/wagon-give-me-a-sign/master/asl/{goal.lower()}.png"
    hint_image.image(image_path, width=200)

    # Stream
    ctx1 = webrtc_streamer(
        key="learn",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_frame_callback=video_frame_callback,
        )

    result_text = st.empty()
    result_text.write(f"👆 Click to start learning")


    while ctx1.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue

        pred = process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[1]

        if pred is None:
            continue

        image_path = f"https://raw.githubusercontent.com/tloizel/wagon-give-me-a-sign/master/asl/{goal.lower()}.png"
        hint_image.image(image_path, width=200)
        result_text.write("")
        predictions_list.append(pred)
        counter += 1
        if counter == 10:
            letter = most_common(predictions_list)
            predictions_list = []
            counter = 0
            if letter == goal :
                goal = random_letter()
                goal_text.write(f"Show us the letter **{goal}**   👉")



if __name__ == "__main__":
    main()
