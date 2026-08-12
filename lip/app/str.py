import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import subprocess
from pathlib import Path

from utils import load_data, num_to_char
from modelutil import load_model

# Set layout
st.set_page_config(layout='wide')

# Centered title
st.markdown("<h2 style='text-align: center;'>🧑‍🏫🧠 Lip Reading App 🎬🗣️</h2>", unsafe_allow_html=True)

# === Webcam Recorder ===
def record_from_webcam(save_path='webcam_input.mpg', duration=3, fps=25, width=100, height=50):
    temp_avi = 'temp.avi'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
        return None

    st.info("📹 Recording for 3 seconds... Please speak clearly.")
    out = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)

    for _ in range(fps * duration):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)

    cap.release()
    out.release()

    # Convert to .mpg
    subprocess.call([
        "ffmpeg", "-y", "-i", temp_avi,
        "-vf", f"fps={fps},scale={width}:{height},format=gray",
        save_path
    ])
    os.remove(temp_avi)
    return save_path

# === User Input Options ===
st.markdown("#### 🎥 Choose Input Method")

use_webcam = st.button("📸 Record from Webcam (3 sec)")
uploaded_file = st.file_uploader("📤 Or upload a .mpg video", type=["mpg"])

selected_video_path = None

if use_webcam:
    selected_video_path = record_from_webcam()
    if selected_video_path:
        st.success("✅ Webcam recording complete.")
elif uploaded_file is not None:
    with open("uploaded_video.mpg", "wb") as f:
        f.write(uploaded_file.read())
    selected_video_path = "uploaded_video.mpg"
    st.success("✅ File uploaded successfully.")
else:
    video_dir = os.path.join('..', 'data', 's1')
    options = os.listdir(video_dir)
    selected_video = st.selectbox("📁 Or choose a sample video:", options)
    selected_video_path = os.path.join(video_dir, selected_video)

# === Process Selected Video ===
if selected_video_path:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎞️ Input Video (autoplay)")

        # Convert and mute using ffmpeg
        converted_video_path = Path("test_video.mp4")
        os.system(f'ffmpeg -y -i "{selected_video_path}" -an -vcodec libx264 -pix_fmt yuv420p {converted_video_path}')

        if converted_video_path.exists():
            st.markdown(
                f"""
                <video autoplay muted controls style="width: 100%; border-radius: 8px; max-height: 200px;">
                    <source src="{converted_video_path.as_posix()}" type="video/mp4">
                </video>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("❌ Failed to convert video.")

    with col2:
        st.markdown("#### 🧠 What the model 'sees' (GIF)")
        video, annotations = load_data(tf.convert_to_tensor(selected_video_path))

        gif_frames = []
        for i, frame in enumerate(video[:40]):
            frame = tf.squeeze(frame)
            frame = tf.image.convert_image_dtype(frame, dtype=tf.uint8)
            frame_np = frame.numpy()
            frame_resized = tf.image.resize(frame_np[..., tf.newaxis], (80, 80)).numpy().astype(np.uint8)
            gif_frames.append(np.squeeze(frame_resized))

        imageio.mimsave('animation.gif', gif_frames, fps=10)
        st.image('animation.gif', width=300)

    # === Prediction ===
    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

    # === Output ===
    st.markdown("#### 📝 Final Lip Reading Result")
    st.markdown(
        f"<div style='background-color:#f0f2f6; padding:10px; border-radius:8px; text-align:center; font-size:18px; font-weight:bold;'>{converted_prediction}</div>",
        unsafe_allow_html=True
    )
