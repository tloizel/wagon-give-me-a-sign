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
from game import random_letter
from twilio_server import get_ice_servers


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
    img = process(img)[0]
    stream = av.VideoFrame.from_ndarray(img, format="bgr24")
    return stream

def process(image):

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
                # answer = f"{ALPHABET[max_index].capitalize()} ({round(max_value,2)*100}%)"
                answer = res[0].capitalize()

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

    return image, answer


def most_common(lst):
    return max(set(lst), key=lst.count)


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
    win = 3


    # Streamlit UI
    st.title("Fingerspelling 🤌")

    st.write("You're here to play 😏")

    start_time = time.time()  # Record the start time

    bar = st.progress(0)
    score_text = st.empty()
    score_text.text(f"Fastest time to get {win} letters wins 🏆")

    goal_text = st.empty()
    goal_text.text(f"Show us the letter {goal}")


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
    result_text.text(f"👆 Click to start the clock")


    while ctx2.state.playing and score < win:
        with lock:
            img = img_container["img"]
        if img is None:
            continue

        pred = process(img)[1]

        if pred is None:
            continue


        result_text.text("")
        score_text.text(f"Score {score}")
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
                goal_text.text(f"Show us the letter {goal}")
                if score == win:
                    st.balloons()
                    end_time = time.time()  # Record the end time
                    time_taken = round(end_time - start_time, 2)  # Calculate the time taken
                    score_text.text(f'It took you {time_taken} seconds. Not bad 👌')
                    goal_text.text(f"You can do better though 🙃")
                    result_text.text(f"👆 Click to play again")


if __name__ == "__main__":
    main()
