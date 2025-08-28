import cv2
import os
import numpy as np
import pickle

# ====================================
# 1️⃣ Capture Face Function
# ====================================
def capture_face(name):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if not os.path.exists("faces_dataset"):
        os.makedirs("faces_dataset")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_name = f"faces_dataset/{name}_{count}.jpg"
            cv2.imwrite(file_name, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("📸 Capturing Faces - Press Q to Stop", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ====================================
# 2️⃣ Train Model Function
# ====================================
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

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save("face_recognizer.yml")
    with open("labels.pkl", "wb") as f:
        pickle.dump(label_map, f)


# ====================================
# 3️⃣ Predict Realtime
# ====================================
def predict_realtime():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer.yml")

    with open("labels.pkl", "rb") as f:
        label_map = pickle.load(f)

    reverse_label_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
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

        cv2.imshow("🎥 Face Recognition - Press Q to Exit", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
