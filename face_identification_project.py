import streamlit as st
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ✅ Fix: Initialize session state
if "name" not in st.session_state:
    st.session_state["name"] = ""

# ---------- Page Config ----------
st.set_page_config(page_title="Face Identification", layout="wide")

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: black;
    }
    .name-box {
        background-color: black;
        color: white;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load Haarcascade ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Video Processor ----------
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.recognizer = None
        self.model_ready = False
        if os.path.exists("trainer.yml"):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read("trainer.yml")
            self.model_ready = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if self.model_ready:
                id_, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 60:
                    name = f"User {id_}"
                else:
                    name = "Unknown"
            else:
                name = st.session_state["name"] if st.session_state["name"] else "Unknown"

            # 🔥 Name box with black background + white text
            cv2.rectangle(img, (x, y-25), (x+w, y), (0, 0, 0), -1)
            cv2.putText(img, name, (x+5, y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

# ---------- Sidebar Navigation ----------
st.sidebar.title("Navigation")
tabs = ["Capture Face", "Train Model", "Predict"]
choice = st.sidebar.radio("Go to", tabs)

# ---------- Capture Tab ----------
if choice == "Capture Face":
    st.header("📸 Capture Face")
    st.session_state["name"] = st.text_input("Enter Name:", st.session_state["name"])

    if st.button("Start Capture"):
        webrtc_streamer(key="capture", video_processor_factory=FaceProcessor)

# ---------- Train Tab ----------
elif choice == "Train Model":
    st.header("🧑‍🏫 Train Model")

    if st.button("Train Now"):
        data_dir = "dataset"
        faces, ids = [], []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith("jpg") or file.endswith("png"):
                    path = os.path.join(root, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    id_ = int(os.path.basename(root).split("_")[-1]) if "_" in os.path.basename(root) else 0
                    faces.append(img)
                    ids.append(id_)
        if faces:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            recognizer.save("trainer.yml")
            st.success("✅ Model Trained & Saved!")
        else:
            st.error("❌ No faces found for training!")

# ---------- Predict Tab ----------
elif choice == "Predict":
    st.header("🔮 Predict Faces")
    webrtc_streamer(key="predict", video_processor_factory=FaceProcessor)
