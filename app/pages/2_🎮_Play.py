import streamlit as st
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from threading import Lock
from time import time

from sys import path
path.append("./")  # Add the root directory to the Python path
from registry import load_model
from game import random_letter
from twilio_server import get_ice_servers
from image_processing import process, most_common


lock2 = Lock()
img_container2 = {"img": None}


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
    return load_model(ml=True, model_name='SVC(probability=True)')

# model = load_model(ml=True, model_name='random_forest_1')
# model = load_model_ml()
model = patience_while_i_load_the_model()


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock2:
        img_container2["img"] = img
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
    goal = random_letter()
    predictions_list = []
    counter = 0
    score = 0
    win = 10


    # Streamlit UI
    st.title("Play 🎮")
    st.write("Let the fastest fingers win")

    st.write("")
    st.write("")

    start_time = time()  # Record the start time

    bar = st.progress(0)
    score_text = st.empty()
    score_text.write(f"Fastest time to get {win} letters wins 🏆")

    st.write("")
    st.write("")

    goal_text = st.empty()
    goal_text.write(f"Show us the letter 👀")


    # Stream
    ctx2 = webrtc_streamer(
        key="play",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_frame_callback=video_frame_callback,
        )

    result_text = st.empty()
    result_text.write(f"👆 Click to start the clock")


    while ctx2.state.playing and score < win:
        with lock2:
            img = img_container2["img"]
        if img is None:
            continue

        pred = process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[1]

        goal_text.write(f"Show us the letter **{goal}**")
        result_text.write("")
        score_text.write(f"Score {score}")
        predictions_list.append(pred)
        counter += 1
        if counter == 10:
            letter = most_common(predictions_list)
            predictions_list = []
            counter = 0
            if letter == goal :
                goal = random_letter()
                score += 1
                bar.progress(round(score*100/win))
                goal_text.write(f"Show us the letter **{goal}**")
                if score == win:
                    st.balloons()
                    end_time = time()  # Record the end time
                    time_taken = round(end_time - start_time, 2)  # Calculate the time taken
                    score_text.write(f'It took you {time_taken} seconds. Not bad 👌')
                    goal_text.write(f"You can do better though 🙃")
                    result_text.write(f"👆 Click to play again")


if __name__ == "__main__":
    main()
