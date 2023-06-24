import streamlit as st
import cv2
import mediapipe as mp
from model import load_model, predict_model_ml
import pandas as pd
from data_proc import preproc_predict
#from tensorflow.keras.models import load_model
import string
ALPHABET = list(string.ascii_lowercase)
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from game import random_letter
import ipdb



def process(image, mp_drawing, mp_drawing_styles, mp_hands, hands, model):

    # global current_prediction
    # global matching_frames
    # global predicting_letter

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
            # predicting_letter = res[0].capitalize()

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


    # print(matching_frames)
    # # Make a prediction for the current frame
    # if current_prediction == predicting_letter and predicting_letter is not None:
    #     matching_frames += 1
    #     if matching_frames == num_consecutive_frames:
    #         current_prediction = random_letter()
    # else:
    #     matching_frames = 0


    # return cv2.flip(image, 1) #check if impact on results
    return image


class VideoProcessor:
    def __init__(self, mp_drawing, mp_drawing_styles, mp_hands, hands, model):
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.mp_hands = mp_hands
        self.hands = hands
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img, self.mp_drawing, self.mp_drawing_styles, self.mp_hands, self.hands, self.model)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():

    RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7)


    # # Initialize variables
    # matching_frames = 0
    # current_prediction = random_letter()
    # predicting_letter = 'helllp'
    # # Set the number of consecutive frames to check
    # num_consecutive_frames = 10


    #load model
    model = load_model()

    video_processor = VideoProcessor(mp_drawing, mp_drawing_styles, mp_hands, hands, model)


    webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=lambda: video_processor,
    async_processing=True,
)





######## UI ##########


# st.text(f'letter to predict is : {current_prediction}')
# st.text(f'number of correct consecutive frames : {matching_frames}')
# st.text(f'predicting letter : {predicting_letter}')






if __name__ == "__main__":
    main()
