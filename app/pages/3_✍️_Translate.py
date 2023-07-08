import streamlit as st
import mediapipe as mp
# from model import load_model_ml, predict_model_ml
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from threading import Lock

import sys
sys.path.append("./")  # Add the root directory to the Python path
from registry import load_model
from image_processing import process, most_common
from twilio_server import get_ice_servers


lock3 = Lock()
img_container3 = {"img": None}


@st.cache_resource(ttl=3600, max_entries=100)
def define_hands():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7)
    return mp_drawing, mp_drawing_styles, mp_hands, hands

mp_drawing, mp_drawing_styles, mp_hands, hands = define_hands()


@st.cache_data(ttl=3600, max_entries=100)
def patience_while_i_load_the_model():
    # Load and return the model
    # return load_model(ml=True, model_name='random_forest_1')
    return load_model(ml=True, model_name='RandomForestClassifier()')

# model = load_model(ml=True, model_name='random_forest_1')
# model = load_model_ml()
model = patience_while_i_load_the_model()


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock3:
        img_container3["img"] = img
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
    predictions_list = []
    counter = 0


    # Streamlit UI
    st.title("Fingerspelling 🤌")

    st.text("Let your fingers do the talking")

    speed = st.slider('Select your speed from 🐌 to ⚡️', 0, 100, 60, step=10)

    # Stream
    ctx3 = webrtc_streamer(
        key="translate",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_frame_callback=video_frame_callback,
        )

    result_text = st.empty()
    result_text.write(f"👆 Click to start translating")

    translation = st.empty()
    sentence = []

    while ctx3.state.playing:
        speed = st.empty()

        with lock3:
            img = img_container3["img"]
        if img is None:
            continue

        pred = process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[1]

        speed.write("")
        result_text.write("")
        predictions_list.append(pred)
        counter += 1
        if counter == int(110 - speed):
            letter = most_common(predictions_list)
            if letter != "No letter":
                sentence.append(letter)
            translation.write(f'{"".join(sentence)}')
            counter = 0
            predictions_list = []


if __name__ == "__main__":
    main()
