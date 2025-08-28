import streamlit as st
import cv2
import os
import numpy as np
import pickle
from datetime import datetime
import base64

st.set_page_config(page_title="face identification", layout="wide") 

# Encode your local image to base64
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

# 👇 Background image
set_bg_local("face_identification_2.jpg")

# Title
st.markdown("""
    <div style="background-color:black;border-radius:10px;text-align:center;">
        <h1 style="color:burlywood;">FACE IDENTIFICATION</h1>
    </div>
""", unsafe_allow_html=True)

# ---------- Sidebar Black Background + White Text ----------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: black !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;   /* Sidebar ke andar sab white */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.sidebar.image("rishu.jpg")
    st.sidebar.header("💬CONTACT US")
    st.sidebar.text("📞 8809972414")
    st.sidebar.text("✉️ rishabhverma190388099@gmail.com")

    st.sidebar.header("👥 About US")
    st.sidebar.text("We are a group of ML engineers working on face Identification")

# ---------- Button Color CSS ----------
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ff6666;   /* हल्का red */
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1em;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff4d4d;   /* hover पर गहरा red */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Text Input CSS (Black background + White text) ----------
st.markdown("""
    <style>
    .stTextInput input {
        width: 100%;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        text-align: left;
        border-radius: 10px;
        border: none;
        background: black;      /* Black background */
        color: white;           /* White text */
    }
    .stTextInput input::placeholder {
        color: #cccccc;         /* हल्का grey placeholder */
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Session States ----------
if "capturing" not in st.session_state:
    st.session_state.capturing = False
if "recognizing" not in st.session_state:
    st.session_state.recognizing = False
if "name" not in st.session_state:
    st.session_state.name = ""

# =========================
# 1️⃣ Capture Face Section
# =========================
def capture_faces(name):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    stframe = st.empty()
    count = 0

    if not os.path.exists("faces_dataset"):
        os.makedirs("faces_dataset")

    while st.session_state.capturing:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_name = f"faces_dataset/{name}_{count}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(file_name, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

# =========================
# 2️⃣ Train Model Section
# =========================
def train_model():
    faces, labels = [], []
    label_map = {}
    current_id = 0

    if not os.path.exists("faces_dataset"):
        st.error("No dataset found! Capture faces first.")
        return

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

# =========================
# 3️⃣ Predict Realtime
# =========================
def predict_realtime():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer.yml")

    with open("labels.pkl", "rb") as f:
        label_map = pickle.load(f)

    reverse_label_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    stframe = st.empty()

    while st.session_state.recognizing:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)
            name = reverse_label_map.get(id_, "Unknown")

            cv2.putText(frame, f"{name} ({int(conf)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()


# =========================
# 🔥 Tabs instead of selectbox
# =========================
tab1, tab2, tab3 = st.tabs(["📸 Capture Face", "📚 Train Model", "🔍 Predict"])

with tab1:
    st.session_state.name = st.text_input("Enter Name:", st.session_state.name)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Capture", key="start_capture"):
            if st.session_state.name.strip() == "":
                st.error("Please enter a name!")
            else:
                st.session_state.capturing = True
                capture_faces(st.session_state.name)

    with col2:
        if st.button("⏹ End Capture", key="end_capture"):
            st.session_state.capturing = False
            st.success("Capture stopped.")

with tab2:
    if st.button("Train", key="train_btn"):
        train_model()
        st.success("✅ Model trained successfully!")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Recognition", key="start_predict"):
            st.session_state.recognizing = True
            predict_realtime()

    with col2:
        if st.button("⏹ Stop Recognition", key="stop_predict"):
            st.session_state.recognizing = False
            st.success("Recognition stopped.")
