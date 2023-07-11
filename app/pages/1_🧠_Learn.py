import streamlit as st
# import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from threading import Lock

from sys import path
path.append("./")  # dd the root directory to the Python path
from game import random_letter
from twilio_server import get_ice_servers
from image_processing import image_process, most_common, define_hands, patience_while_i_load_the_model


lock1 = Lock()
img_container1 = {"img": None}

if 'mp_drawing' not in st.session_state:
    st.session_state['mp_drawing'], st.session_state['mp_drawing_styles'], st.session_state['mp_hands'], st.session_state['hands'] = define_hands()

mp_drawing = st.session_state['mp_drawing']
mp_drawing_styles = st.session_state['mp_drawing_styles']
mp_hands = st.session_state['mp_hands']
hands = st.session_state['hands']

# mp_drawing, mp_drawing_styles, mp_hands, hands = define_hands()

# model = load_model(ml=True, model_name='random_forest_1')
# model = load_model_ml()
model = patience_while_i_load_the_model()


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock1:
        img_container1["img"] = img
    img = image_process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[0]
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
        with lock1:
            img = img_container1["img"]
        if img is None:
            continue

        pred = image_process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[1]

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
