import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import io
import plotly.graph_objects as go
import time
import base64
from datetime import datetime
from pydub import AudioSegment
import warnings

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Audio Deepfake Detection",
    page_icon="🔊",
    layout="centered"
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("updated_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ------------------------------
# Audio Feature Extraction
# ------------------------------
def extract_features_from_audio(audio_bytes, file_type, max_length=500, sr=16000, n_mfcc=40):
    try:

        # Handle M4A files
        if file_type == "m4a":
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="m4a"
            )

            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)

            audio_array, _ = librosa.load(wav_io, sr=sr)

        else:
            audio_array, _ = librosa.load(
                io.BytesIO(audio_bytes),
                sr=sr
            )

        mfccs = librosa.feature.mfcc(
            y=audio_array,
            sr=sr,
            n_mfcc=n_mfcc
        )

        # Padding / Trimming
        if mfccs.shape[1] < max_length:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(
                mfccs,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )
        else:
            mfccs = mfccs[:, :max_length]

        mfccs = mfccs.reshape(
            1,
            mfccs.shape[0],
            mfccs.shape[1],
            1
        )

        return mfccs, audio_array, sr

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# ------------------------------
# Waveform Plot
# ------------------------------
def create_waveform(audio_data, sr):

    time_axis = np.linspace(
        0,
        len(audio_data) / sr,
        len(audio_data)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform'
    ))

    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        height=300
    )

    return fig

# ------------------------------
# Confidence Gauge
# ------------------------------
def create_gauge(confidence):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "red"}
            ]
        }
    ))

    fig.update_layout(height=300)

    return fig

# ------------------------------
# HTML Report Generator
# ------------------------------
def generate_html_report(result):

    status = "DEEPFAKE DETECTED" if result['is_deepfake'] else "AUTHENTIC AUDIO"

    html = f"""
    <html>
    <head>
        <title>Audio Deepfake Report</title>
    </head>

    <body style="font-family: Arial; padding: 30px;">

        <h1>🔊 Audio Deepfake Detection Report</h1>

        <hr>

        <h2>Result: {status}</h2>

        <h3>Confidence: {result['confidence']*100:.2f}%</h3>

        <p><strong>File Name:</strong> {result['file_name']}</p>

        <p><strong>Analysis Time:</strong> {datetime.now()}</p>

    </body>
    </html>
    """

    return html

# ------------------------------
# Download Button
# ------------------------------
def download_button(content, filename):

    b64 = base64.b64encode(
        content.encode()
    ).decode()

    href = f'''
    <a href="data:text/html;base64,{b64}"
    download="{filename}">
        Download Report
    </a>
    '''

    return href

# ------------------------------
# UI
# ------------------------------
st.title("🔊 Audio Deepfake Detection")

st.write("Upload an audio file to detect fake audio.")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=['wav', 'mp3', 'ogg', 'flac', 'm4a']
)

# ------------------------------
# Prediction
# ------------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)

    st.write(f"📄 File Name: {uploaded_file.name}")
    st.write(f"💾 File Size: {file_size:.2f} MB")

    if st.button("Analyze Audio"):

        if model is None:
            st.stop()

        with st.spinner("Processing Audio..."):

            file_type = uploaded_file.name.split(".")[-1].lower()

            features, audio_data, sr = extract_features_from_audio(
                uploaded_file.getvalue(),
                file_type
            )

            if features is not None:

                prediction = model.predict(features)

                confidence = float(prediction[0][0])

                is_deepfake = confidence > 0.5

                # Results
                if is_deepfake:
                    st.error("🚨 Deepfake Audio Detected")
                else:
                    st.success("✅ Authentic Audio")

                st.write(f"Confidence: {confidence*100:.2f}%")

                # Gauge
                gauge_fig = create_gauge(confidence)
                st.plotly_chart(
                    gauge_fig,
                    use_container_width=True
                )

                # Waveform
                waveform_fig = create_waveform(
                    audio_data,
                    sr
                )

                st.plotly_chart(
                    waveform_fig,
                    use_container_width=True
                )

                # Save Result
                result = {
                    'confidence': confidence,
                    'is_deepfake': is_deepfake,
                    'file_name': uploaded_file.name
                }

                # Generate Report
                html_report = generate_html_report(result)

                st.markdown(
                    download_button(
                        html_report,
                        "deepfake_report.html"
                    ),
                    unsafe_allow_html=True
                )

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("About")

st.sidebar.info("""
This application detects whether an uploaded audio file is real or AI-generated using Deep Learning.
""")

st.sidebar.markdown("""
### Supported Formats
- WAV
- MP3
- OGG
- FLAC
- M4A
""")
