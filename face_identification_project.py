import streamlit as st
import cv2
import av
import os
import numpy as np
import pickle
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading

# ---------- Page config ----------
st.set_page_config(page_title="Face Identification", layout="wide")

# ---------- CSS ----------
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
    unsafe_allow_html=True,
)

# ---------- Background (optional) ----------
def set_bg_local(image_file):
    if os.path.exists(image_file):
        import base64
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
            unsafe_allow_html=True,
        )


set_bg_local("face_identification_2.jpg")

# ---------- Title ----------
st.markdown(
    """
    <div style="background-color:black;border-radius:10px;text-align:center;padding:8px;">
        <h1 style="color:burlywood;margin:0;">FACE IDENTIFICATION</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    if os.path.exists("rishu.jpg"):
        st.image("rishu.jpg")
    st.header("💬 CONTACT US")
    st.text("📞 8809972414")
    st.text("✉️ rishabhverma190388099@gmail.com")
    st.header("👥 About US")
    st.text("We are a group of ML engineers working on face Identification")

# ---------- Globals ----------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
DATA_DIR = "faces_dataset"
MODEL_PATH = "face_recognizer.yml"
LABELS_PATH = "labels.pkl"
os.makedirs(DATA_DIR, exist_ok=True)

# Lock for safe writes
write_lock = threading.Lock()

# ---------- Helper: train function (called from main thread) ----------
def train_model():
    faces, labels = [], []
    label_map = {}
    current_id = 0

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")]
    if len(files) < 2:
        st.error("Not enough data to train! Capture more faces (min 2 images).")
        return False

    for file in files:
        path = os.path.join(DATA_DIR, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        name = file.split("_")[0]
        if name not in label_map:
            label_map[name] = current_id
            current_id += 1
        faces.append(img)
        labels.append(label_map[name])

    if len(faces) < 2:
        st.error("Not enough valid images to train.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    return True

# ---------- Video Processors (use recv() API) ----------

class FaceCaptureProcessor(VideoProcessorBase):
    """
    Saves detected faces to DATA_DIR every `save_every` processed frames.
    `name` is passed when webrtc_streamer is started (so avoid st.* inside).
    """
    def __init__(self, name: str, save_every: int = 5):
        self.name = name or "unknown"
        self.save_every = max(1, int(save_every))
        self.frame_count = 0
        self.save_count = 0
        # use smaller processing size for speed
        self.proc_width = 320
        self.proc_height = 240

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to ndarray
        img = frame.to_ndarray(format="bgr24")

        # Resize for faster detection (we will display the resized frame)
        small = cv2.resize(img, (self.proc_width, self.proc_height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        # Only detect on some frames to save CPU
        if self.frame_count % 2 == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
            for (x, y, w, h) in faces:
                # Save only every `save_every` processed frames
                if self.frame_count % self.save_every == 0:
                    # scale back coords are not needed because we save the detected crop from small
                    face_crop = gray[y:y+h, x:x+w]
                    # Write to disk thread-safely
                    with write_lock:
                        self.save_count += 1
                        fname = os.path.join(
                            DATA_DIR,
                            f"{self.name}_{self.save_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
                        )
                        cv2.imwrite(fname, face_crop)

                # Draw rectangle on small frame
                cv2.rectangle(small, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # return the smaller frame (faster to transfer)
        return av.VideoFrame.from_ndarray(small, format="bgr24")


class FaceRecognitionProcessor(VideoProcessorBase):
    """
    Performs recognition using LBPH model if available.
    If model missing, draws boxes and shows 'Train model' label.
    """
    def __init__(self):
        self.proc_width = 320
        self.proc_height = 240
        self.frame_count = 0
        # load model if present
        self.model_ready = False
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(MODEL_PATH)
                with open(LABELS_PATH, "rb") as f:
                    label_map = pickle.load(f)
                self.reverse_map = {v: k for k, v in label_map.items()}
                self.model_ready = True
            except Exception:
                self.model_ready = False
        else:
            self.model_ready = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img, (self.proc_width, self.proc_height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        # process every 3rd frame for speed
        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(small, format="bgr24")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

        for (x, y, w, h) in faces:
            if self.model_ready:
                try:
                    id_, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                    name = self.reverse_map.get(id_, "Unknown")
                except Exception:
                    name = "Unknown"
                    conf = 0
                label = f"{name} ({int(conf)})"
            else:
                label = "Train model"

            # black rectangle behind text
            text_bg_top = max(0, y - 30)
            cv2.rectangle(small, (x, text_bg_top), (x + w, y), (0, 0, 0), -1)
            # white text
            cv2.putText(small, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # bounding box
            cv2.rectangle(small, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(small, format="bgr24")

# ---------- UI: Tabs ----------
tab1, tab2, tab3 = st.tabs(["📸 Capture Face (Realtime)", "📚 Train Model", "🔍 Predict (Realtime)"])

# CAPTURE tab: require entering name before starting capture
with tab1:
    name_input = st.text_input("Enter Name (used for saved images):", value=st.session_state.get("name", ""))
    start_capture = st.button("Start Capture Stream")
    stop_capture = st.button("Stop Capture Stream")

    st.session_state["name"] = name_input.strip()

    if start_capture and st.session_state["name"] != "":
        # start webrtc streamer with capture processor that knows the name
        webrtc_streamer(
            key="capture",
            video_processor_factory=lambda: FaceCaptureProcessor(name=st.session_state["name"], save_every=6),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    elif stop_capture:
        # stopping is handled by the UI stop button inside stream; webrtc_streamer creates its own controls.
        st.write("If stream is running, click STOP (red) in the player to stop it.")

# TRAIN tab
with tab2:
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            ok = train_model()
        if ok:
            st.success("✅ Model trained and saved.")
        else:
            st.error("Training failed or not enough data.")

# PREDICT tab
with tab3:
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        webrtc_streamer(
            key="recognition",
            video_processor_factory=FaceRecognitionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.warning("Model not found. Please capture faces and train the model first.")
