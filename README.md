# 🎤 Speech Emotion Recognition

An AI-powered **Speech Emotion Recognition (SER)** web application that analyzes human speech and predicts the emotion expressed in the audio.

The project uses **Machine Learning, MFCC feature extraction, SVM classification, and Flask** to provide real-time emotion prediction through a simple web interface.

## 🌐 Live Demo

🚀 **Try the application:**
https://speech-emotion-recognition-6iox.onrender.com

## 📌 Project Overview

Speech Emotion Recognition is a machine learning application designed to identify emotions from human speech.

The system takes a `.wav` audio file as input, extracts meaningful audio features using **MFCC (Mel-Frequency Cepstral Coefficients)**, and uses a trained **Support Vector Machine (SVM)** model to predict the emotion.

### 🎯 Workflow

```text
Audio File
    ↓
Audio Preprocessing
    ↓
MFCC Feature Extraction
    ↓
Feature Scaling
    ↓
SVM Machine Learning Model
    ↓
Emotion Prediction
    ↓
Confidence & Probability Results
```

## ✨ Features

* 🎤 Upload `.wav` audio files
* 🧠 Machine Learning-based emotion prediction
* 🎵 MFCC audio feature extraction
* 📊 Emotion probability visualization
* 📈 Prediction confidence score
* 🌐 Flask web application
* 🚀 Deployed online using Render
* 📦 Model saved using Joblib

## 😊 Emotions

The model is trained to recognize emotions available in the RAVDESS dataset, including:

* Angry
* Calm
* Disgust
* Fearful
* Happy
* Neutral
* Sad
* Surprised

## 🛠️ Technologies Used

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Core programming language            |
| Flask        | Web application backend              |
| Librosa      | Audio processing and MFCC extraction |
| NumPy        | Numerical computation                |
| Pandas       | Dataset processing                   |
| Scikit-learn | Machine Learning                     |
| SVM          | Emotion classification               |
| Joblib       | Model serialization                  |
| HTML         | Frontend structure                   |
| CSS          | Frontend styling                     |
| Git & GitHub | Version control                      |
| Render       | Cloud deployment                     |

## 📂 Project Structure

```text
speech-emotion-recognition/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── features.csv
│
├── models/
│   ├── emotion_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── extract_features.py
│   ├── train_model.py
│   ├── predict_emotion.py
│   ├── evaluate_model.py
│   └── test_audio.py
│
├── static/
│   ├── style.css
│   └── uploads/
│
└── templates/
    ├── index.html
    └── result.html
```

## 📊 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The audio data contains speech samples representing different emotional expressions.

Dataset used for this project:

**RAVDESS 16 kHz Speech Dataset**

The original raw dataset is not included in this repository because of its size.

## 🔍 Feature Extraction

The system extracts **40 MFCC features** from each audio file.

MFCCs are commonly used in speech and audio processing because they represent important characteristics of the human voice.

The extracted features are stored in:

```text
dataset/features.csv
```

## 🤖 Machine Learning Model

The project uses a **Support Vector Machine (SVM)** classifier with an RBF kernel.

Before training:

1. Audio features are extracted.
2. Features are split into training and testing sets.
3. `StandardScaler` is used to normalize the features.
4. The SVM model is trained.
5. The trained model and scaler are saved using Joblib.

Saved files:

```text
models/emotion_model.pkl
models/scaler.pkl
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shivapriyabagari/speech-emotion-recognition.git
```

### 2. Open the project

```bash
cd speech-emotion-recognition
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the Flask application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

## 🚀 Deployment

The application is deployed using **Render**.

Production server:

```text
Gunicorn
```

Start command:

```bash
gunicorn app:app
```

Live application:

**https://speech-emotion-recognition-6iox.onrender.com**

## 🧪 How to Use

1. Open the live application.
2. Click **Choose File**.
3. Select a `.wav` speech file.
4. Click **Analyze Emotion**.
5. The application processes the audio.
6. The predicted emotion is displayed.
7. The application also shows the prediction confidence and emotion probabilities.

## 📈 Future Improvements

Possible future improvements include:

* 🎙️ Direct microphone recording
* ⚡ Real-time emotion detection
* 📱 Responsive mobile interface
* 🤖 Deep learning models such as CNN/LSTM
* 🎧 Support for additional audio formats
* 📊 More detailed emotion analytics
* ☁️ Improved cloud deployment and scalability

## 👩‍💻 Author

**ShivaPriya Bagari**

B.Tech – Computer Science & Engineering

GitHub:
https://github.com/shivapriyabagari

## ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐.

---

**Built with Python, Machine Learning and Flask**
