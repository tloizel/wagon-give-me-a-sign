import streamlit as st
import cv2
import mediapipe as mp
from model import load_model_ml, predict_model_ml
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from game import random_letter
import ipdb
from data_proc import preproc_predict
import threading

lock = threading.Lock()
img_container = {"img": None}

# load hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7)

# load model
model = load_model_ml()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame


def process_out_of_class(image):

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
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

    def process(self, image):

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

        return image

    def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = self.process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def most_common(lst):
    return max(set(lst), key=lst.count)

def main():

    RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Streamlit UI
    st.title("Letter Prediction")
    goal = random_letter()
    goal_text = st.empty()
    goal_text.text(f"# Do a {goal}")


    # Stream
    ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory= VideoProcessor,
        async_processing=True,
        video_frame_callback=video_frame_callback,
        )

    result_text = st.empty()
    result_text.text(f"Let's play")

    congratulations_text = st.empty()

    predictions_list = []
    counter = 0

    while ctx.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue

        pred = process_out_of_class(img)

        if pred is None:
            continue

        result_text.text(f"You are doing a {pred}")

        predictions_list.append(pred)
        counter += 1
        if counter == 10:
            letter = most_common(predictions_list)
            predictions_list = []
            counter = 0
            if letter == goal :
                goal = random_letter()
                congratulations_text.text(f"Yaaay 🍾")
                goal_text.text(f"# Do a {goal}")
            else:
                congratulations_text.text("")






if __name__ == "__main__":
    main()
