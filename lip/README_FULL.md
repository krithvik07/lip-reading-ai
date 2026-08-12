
# 🧠 Lip Reading Inference Using LipNet (Visual Speech Recognition)

This project implements a visual speech recognition system using **LipNet**, one of the earliest end-to-end deep learning architectures for lip reading. The system predicts spoken sentences from silent video input using only visual lip movement, without relying on any audio.

---

## 📄 Abstract

Lip reading, also called visual speech recognition, allows machines to understand speech by analyzing facial movement. This project uses a pretrained LipNet model to predict structured spoken sentences from videos in the GRID corpus. A **Streamlit web app** enables users to upload or select `.mpg` videos and see the model's predictions. The app displays both the color video and the grayscale model input (animated), followed by the predicted output.

The model works best with videos that resemble the GRID dataset's structure (fixed-length, frontal face, specific vocabulary). The project also highlights the limitation of current models in generalizing to real-world speech without extensive retraining.

---

## 🧩 Architecture Diagram

```
                         ┌────────────────────┐
                         │  Video Selection   │
                         └─────────┬──────────┘
                                   │
                         ┌────────▼───────────┐
                         │  FFmpeg Conversion │
                         └────────┬───────────┘
                                   │
                         ┌────────▼────────────┐
                         │ Grayscale Preprocess│
                         └────────┬────────────┘
                                   │
                         ┌────────▼────────────┐
                         │    LipNet Model     │
                         └────────┬────────────┘
                                   │
                        ┌─────────▼─────────────┐
                        │   CTC Decoding Output │
                        └─────────┬─────────────┘
                                  ▼
                         Displayed in Web App
```

---

## 📦 Dataset Used: GRID Corpus

- 34 speakers (25 male, 9 female)
- 1,000 videos per speaker
- Each video is ~3 seconds, 25 FPS, 100x50 grayscale
- Sentence format:  
  `command + color + preposition + letter + digit + adverb`  
  Example: `"bin blue at A 1 soon"`

- Data format:  
  - `.mpg` video files  
  - `.align` ground truth files (used during training)

---

## 🧠 Model Overview: LipNet

- Input shape: `(75 frames, 46 height, 140 width, 1)`
- Model:
  - `Conv3D` layers extract spatiotemporal features
  - `TimeDistributed(Flatten)`
  - `Bidirectional LSTM x2`
  - `Dense` softmax layer for character probabilities
  - **CTC Loss** + **CTC Greedy Decoding**

---

## 🔄 Workflow

1. User selects a video (`.mpg`)
2. Video is converted to `.mp4` for display
3. Frames are extracted and resized to `(140x46)`
4. Grayscale animation (`.gif`) is generated
5. Model predicts character sequences
6. CTC decoding turns sequences into text
7. Final result is shown in the app

---

## 🛠️ Technologies Used

- Python 3.9
- TensorFlow 2.10
- Keras
- OpenCV
- FFmpeg
- Streamlit
- NumPy
- ImageIO
- GRID Dataset

---

## 🧰 Requirements & Installation

### 🔧 Install Dependencies

Create a Python 3.9 virtual environment and install dependencies:

```bash
python -m venv lipnet-env
source lipnet-env/bin/activate  # or .\lipnet-env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 📦 `requirements.txt` Example

```text
tensorflow==2.10.1
numpy==1.23.5
pandas==1.5.3
streamlit==1.21.0
opencv-python-headless==4.7.0.72
imageio==2.31.1
imageio-ffmpeg==0.4.9
Pillow==9.5.0
```

Ensure **FFmpeg** is installed and added to your system path.

---

## 🚀 Running the Project

### Step 1: Place your test videos in:
```
LipNet-main/data/s1/
```

### Step 2: Launch the app

```bash
streamlit run app/streamlitapp.py
```

### Step 3: Interact

- Select a `.mpg` video from the dropdown
- Watch the model predict the sentence visually

---

## 💡 Use Cases

| Application              | Description                                      |
|--------------------------|--------------------------------------------------|
| Accessibility            | Helps hearing-impaired individuals               |
| Silent Command Systems   | For surgery, military, or space communication    |
| Surveillance             | Analyzing muted video feeds                      |
| Noisy Environments       | Where audio speech recognition fails             |

---

## ⚠️ Limitations

| Limitation                      | Cause                                      |
|--------------------------------|--------------------------------------------|
| Low accuracy on real-world data| Model is trained only on GRID              |
| Needs `.align` files to train  | Supervised training requires known labels  |
| Doesn’t generalize well        | Fixed vocabulary, camera angle, lighting   |
| Not suitable for spontaneous speech | Only works on structured GRID sentences  |

---

## ✅ How to Improve / Extend

- 🔁 Retrain LipNet on real-world datasets like **LRS2**, **LRS3**, or **AVSpeech**
- 🤖 Use transformer-based models like **AV-HuBERT**, **TM-seq2seq**, or **VSRNet**
- 🧠 Integrate text-to-speech (TTS) for accessibility
- 📈 Add attention visualization or prediction confidence
- ☁️ Deploy on HuggingFace Spaces or as a cloud API

---

## 🧪 Demo Features

- Dropdown to select GRID `.mpg` files
- Displays color video + model input GIF
- Predicts spoken sentence using visual data only
- Fully interactive web UI using Streamlit

---

## 🧠 Final Note

This project demonstrates **what's possible with visual speech recognition** but also where current limitations lie. It's a strong starting point for academic, assistive, and R&D applications — but needs further training and augmentation to reach real-world robustness.

---
