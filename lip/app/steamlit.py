import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import load_data, num_to_char
from modelutil import load_model

# Set Streamlit layout
st.set_page_config(layout='wide')

# Centered creative title
st.markdown("<h2 style='text-align: center;'>🧑‍🏫🧠 Lip Reading App 🎬🗣️</h2>", unsafe_allow_html=True)

# Load available videos
video_dir = '.'
options = [f for f in os.listdir(video_dir) if f.endswith('.mpg') or f.endswith('.mp4')]
selected_video = st.selectbox('Choose a video to analyze', options)

# Layout columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('Input video (converted to MP4, audio removed)')
        file_path = os.path.join(video_dir, selected_video)

        # Remove audio and convert to MP4
        os.system(f'ffmpeg -i "{file_path}" -an -vcodec libx264 -pix_fmt yuv420p test_video.mp4 -y')

        # Display the muted video
        with open('test_video.mp4', 'rb') as video_file:
            st.video(video_file.read())

    with col2:
        st.info('What the model "sees" (grayscale lip-only input)')

        # Load preprocessed video
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # Prepare GIF frames
        gif_frames = []
        max_frames = 40
        for i, frame in enumerate(video):
            if i >= max_frames:
                break
            frame = tf.squeeze(frame)
            frame = tf.image.convert_image_dtype(frame, dtype=tf.uint8)
            frame_np = frame.numpy()

            # Resize and convert
            frame_resized = tf.image.resize(frame_np[..., tf.newaxis], (80, 80)).numpy().astype(np.uint8)
            gif_frames.append(np.squeeze(frame_resized))

        imageio.mimsave('animation.gif', gif_frames, fps=10)
        st.image('animation.gif', width=300)

        # Inference
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        # Final decoded prediction
        st.info('Final lip reading result')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
