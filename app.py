import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
import tempfile

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Function to extract faces from video
@st.cache_data
def extract_faces_from_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if len(frames) < 10 and results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                head_width = int(w * 1.5)
                head_height = int(h * 1.5)
                x = max(0, x - (head_width - w) // 2)
                y = max(0, y - (head_height - h) // 2)
                head_width = min(iw - x, head_width)
                head_height = min(ih - y, head_height)

                face_image = frame[y:y + head_height, x:x + head_width]
                face_image = cv2.resize(face_image, (80, 80))

                frames.append(face_image)

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()
    return frames

# Function to display classification result
def display_classification_result(Y_lab):
    if Y_lab == 0:
        st.markdown('<h2 style="color:red;font-size:30px;text-align:center;">This Video is Fake</h2>', unsafe_allow_html=True)
    elif Y_lab == 1:
        st.markdown('<h2 style="color:green;font-size:30px;text-align:center;">This Video is Original</h2>', unsafe_allow_html=True)

# Main function
def main():
    # Set page title and header
    # st.set_page_config(page_title="Deep Fake Video Classification", layout="wide")
    st.title("Deep Fake Video Classification")

    # Display upload file widget
    uploaded_file = st.file_uploader("Upload a video...", type=["mp4"])

    if uploaded_file is not None:
        with st.spinner('Processing video...'):
            # Temporarily save the uploaded video
            temp_video_file = tempfile.NamedTemporaryFile(delete=False)
            temp_video_file.write(uploaded_file.getvalue())

            # Extract faces from the uploaded video
            frames = extract_faces_from_video(temp_video_file.name)

            # Load the pre-trained model
            model = tf.keras.models.load_model('final_model_conv_3d')

            # Perform classification
            if frames:
                frames = np.array(frames)
                frames = np.expand_dims(frames, axis=0)  # Add batch dimension
                Y_pred = model.predict(frames)
                Y_lab = np.argmax(Y_pred)

                # Display classification result
                st.subheader("Classification Result")
                display_classification_result(Y_lab)

    # Remove the temporarily saved video file
    if uploaded_file is None:
        try:
            temp_video_file.close()
        except NameError:
            pass

if __name__ == "__main__":
    main()
