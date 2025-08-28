import streamlit as st
import cv2
import os
import numpy as np
import pickle
from datetime import datetime
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading

st.set_page_config(page_title="Face Identification", layout="wide")

# =====================
# Custom CSS (Sidebar Black)
# =====================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: black;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================
# Background function
# =====================
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_local("face_identification_2.jpg")

# =====================
# Title
# =====================
st.markdown("""
    <div style="background-color:black;border-radius:10px;text-align:center;">
        <h1 style="color:burlywood;">FACE IDENTIFICATION</h1>
    </div>
""", unsafe_allow_html=True)

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.image("rishu.jpg")
    st.header("💬 CONTACT US")
    st.text("📞 8809972414")
    st.text("✉️ rishabhverma190388099@gmail.com")
    st.header("👥 About US")
    st.text("We are a group of ML engineers working on face Identification")

# =====================
# Face Capture + Train + Predict
# =====================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if "name" not in st.session_state:
    st.session_state.name = ""

if not os.path.exists("faces_dataset"):
    os.makedirs("faces_dataset")

# Global lock for thread safety
lock = threading.Lock()

# 1️⃣ Capture Faces with WebRTC
class FaceCapture(VideoTransformerBase):
    def __init__(self):
        self.count = 0
        self.frame_skip = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize for speed
        img = cv2.resize(img, (320, 240))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Skip alternate frames
        self.frame_skip += 1
        if self.frame_skip % 2 != 0:
            return img

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            self.count += 1
            face_img = gray[y:y+h, x:x+w]
            file_name = f"faces_dataset/{st.session_state.name}_{self.count}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(file_name, face_img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img


# 2️⃣ Train Model
def train_model():
    faces, labels = [], []
    label_map = {}
    current_id = 0

    for file in os.listdir("faces_dataset"):
        if file.endswith(".jpg"):
            path = os.path.join("faces_dataset", file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            name = file.split("_")[0]

            if name not in label_map:
                label_map[name] = current_id
                current_id += 1

            faces.append(img)
            labels.append(label_map[name])

    if len(faces) < 2:
        st.error("Not enough data to train! Capture more faces.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save("face_recognizer.yml")

    with open("labels.pkl", "wb") as f:
        pickle.dump(label_map, f)


# 3️⃣ Predict Realtime with WebRTC
class FaceRecognition(VideoTransformerBase):
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("face_recognizer.yml")
        with open("labels.pkl", "rb") as f:
            self.label_map = pickle.load(f)
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.frame_skip = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize for smoothness
        img = cv2.resize(img, (320, 240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Skip alternate frames
        self.frame_skip += 1
        if self.frame_skip % 2 != 0:
            return img

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = self.recognizer.predict(roi_gray)
            name = self.reverse_map.get(id_, "Unknown")

            # 🔥 Black background rectangle behind name
            cv2.rectangle(img, (x, y-40), (x+w, y), (0, 0, 0), -1)

            # 🔥 White text on black
            cv2.putText(img, f"{name} ({int(conf)})", (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 🔵 Face bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img


# =====================
# Tabs
# =====================
tab1, tab2, tab3 = st.tabs(["📸 Capture Face", "📚 Train Model", "🔍 Predict"])

with tab1:
    st.session_state.name = st.text_input("Enter Name:", st.session_state.name)
    if st.session_state.name.strip() != "":
        webrtc_streamer(
            key="capture",
            video_transformer_factory=FaceCapture,
            media_stream_constraints={"video": True, "audio": False}
        )
    else:
        st.error("Please enter a name before capturing.")

with tab2:
    if st.button("Train Model"):
        train_model()
        st.success("✅ Model trained successfully!")

with tab3:
    webrtc_streamer(
        key="recognition",
        video_transformer_factory=FaceRecognition,
        media_stream_constraints={"video": True, "audio": False}
    )
