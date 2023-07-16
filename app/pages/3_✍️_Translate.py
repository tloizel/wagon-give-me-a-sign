import streamlit as st
# import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from threading import Lock

from sys import path
path.append("./")  # Add the root directory to the Python path
from twilio_server import get_ice_servers
from image_processing import image_process, most_common, define_hands, patience_while_i_load_the_model


lock3 = Lock()
img_container3 = {"img": None}

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
    with lock3:
        img_container3["img"] = img
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
    predictions_list = []
    counter = 0


    # Streamlit UI
    st.title("Translate ✍️")

    st.write("Let your fingers do the talking")

    st.write("")

    with st.expander("Some tips 👀"):
        st.write("🖐️ for space")
        st.write("👍 (to the left) to delete")

    st.write("")
    speed = st.slider('Select your speed from 🐌 to ⚡️', 0, 100, 50, step=10)


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

    translation = st.markdown(
     f"""
    <div style="background-color: #ffcc0336; padding: 10px">
        <p style="margin:0px;">Your text will appear here 👀</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    sentence = []

    while ctx3.state.playing:

        with lock3:
            img = img_container3["img"]
        if img is None:
            continue

        pred = image_process(img, mp_drawing, mp_drawing_styles, mp_hands, hands, model)[1]

        result_text.write("Writing 👇")
        predictions_list.append(pred)
        counter += 1
        if counter == int(110 - speed):
        # if counter == 60:
            letter = most_common(predictions_list)
            if letter == "Delete" and len(sentence) >=1 :
                sentence.pop()
            elif letter == "Space":
                sentence.append(" ")
            elif letter != "No letter":
                sentence.append(letter)

            if sentence == []:
                translation.markdown(
                f"""
                <div style="background-color: #ffcc0336; padding: 10px">
                    <p style="margin:0px;">Your text will appear here 👀</p>
                </div>
                """,
                unsafe_allow_html=True
                )
            else:
                translation.markdown(
                f"""
                <div style="background-color: #ffcc0336; padding: 10px">
                    <p style="margin:0px;">{"".join(sentence).capitalize()}</p>
                </div>
                """,
                unsafe_allow_html=True
                )
            # translation.write(f'{"".join(sentence).capitalize()}')
            counter = 0
            predictions_list = []


if __name__ == "__main__":
    main()
