import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import io
import plotly.graph_objects as go
import base64
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Audio Deepfake Detection",
    page_icon="🔊",
    layout="centered"
)

# ---------------------------------------------------
# SIMPLE CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#1E88E5;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.stButton button {
    width:100%;
    background:#1E88E5;
    color:white;
    border:none;
    padding:12px;
    border-radius:10px;
    font-size:18px;
}

.stButton button:hover {
    background:#1565C0;
    color:white;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
    margin-top:20px;
}

.real {
    background:#E8F5E9;
    color:#2E7D32;
}

.fake {
    background:#FFEBEE;
    color:#C62828;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("updated_model.h5")
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------------------------------------------
# AUDIO FEATURE EXTRACTION
# ---------------------------------------------------
def extract_features_from_audio(
    audio_bytes,
    max_length=500,
    sr=16000,
    n_mfcc=40
):

    try:

        audio_array, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=sr,
            mono=True
        )

        mfccs = librosa.feature.mfcc(
            y=audio_array,
            sr=sr,
            n_mfcc=n_mfcc
        )

        # Padding or trimming
        if mfccs.shape[1] < max_length:

            pad_width = max_length - mfccs.shape[1]

            mfccs = np.pad(
                mfccs,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )

        else:
            mfccs = mfccs[:, :max_length]

        # Reshape
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

# ---------------------------------------------------
# WAVEFORM
# ---------------------------------------------------
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

# ---------------------------------------------------
# CONFIDENCE GAUGE
# ---------------------------------------------------
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

# ---------------------------------------------------
# HTML REPORT
# ---------------------------------------------------
def generate_html_report(result):

    status = (
        "DEEPFAKE DETECTED"
        if result['is_deepfake']
        else "AUTHENTIC AUDIO"
    )

    html = f"""
    <html>

    <head>
        <title>Audio Deepfake Report</title>
    </head>

    <body style="font-family:Arial;padding:30px;">

        <h1>🔊 Audio Deepfake Detection Report</h1>

        <hr>

        <h2>Result: {status}</h2>

        <h3>Confidence: {result['confidence']*100:.2f}%</h3>

        <p><strong>File Name:</strong>
        {result['file_name']}</p>

        <p><strong>Generated Time:</strong>
        {datetime.now()}</p>

    </body>

    </html>
    """

    return html

# ---------------------------------------------------
# DOWNLOAD BUTTON
# ---------------------------------------------------
def download_button(content, filename):

    b64 = base64.b64encode(
        content.encode()
    ).decode()

    href = f"""
    <a href="data:text/html;base64,{b64}"
    download="{filename}">
        Download HTML Report
    </a>
    """

    return href

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------
st.markdown(
    '<div class="main-title">🔊 Audio Deepfake Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload audio to detect fake audio</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=['wav', 'mp3', 'ogg', 'flac', 'm4a']
)

# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    file_size = (
        len(uploaded_file.getvalue()) / (1024 * 1024)
    )

    st.write(f"📄 File Name: {uploaded_file.name}")
    st.write(f"💾 File Size: {file_size:.2f} MB")

    if st.button("Analyze Audio"):

        if model is None:
            st.stop()

        with st.spinner("Analyzing Audio..."):

            features, audio_data, sr = (
                extract_features_from_audio(
                    uploaded_file.getvalue()
                )
            )

            if features is not None:

                prediction = model.predict(
                    features,
                    verbose=0
                )

                confidence = float(prediction[0][0])

                is_deepfake = confidence > 0.5

                # ---------------------------------------------------
                # RESULT
                # ---------------------------------------------------
                if is_deepfake:

                    st.markdown(
                        '<div class="result-box fake">'
                        '🚨 Deepfake Audio Detected'
                        '</div>',
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        '<div class="result-box real">'
                        '✅ Authentic Audio'
                        '</div>',
                        unsafe_allow_html=True
                    )

                st.write(
                    f"### Confidence: {confidence*100:.2f}%"
                )

                # ---------------------------------------------------
                # GAUGE
                # ---------------------------------------------------
                gauge_fig = create_gauge(confidence)

                st.plotly_chart(
                    gauge_fig,
                    use_container_width=True
                )

                # ---------------------------------------------------
                # WAVEFORM
                # ---------------------------------------------------
                waveform_fig = create_waveform(
                    audio_data,
                    sr
                )

                st.plotly_chart(
                    waveform_fig,
                    use_container_width=True
                )

                # ---------------------------------------------------
                # SAVE RESULT
                # ---------------------------------------------------
                result = {
                    'confidence': confidence,
                    'is_deepfake': is_deepfake,
                    'file_name': uploaded_file.name
                }

                # ---------------------------------------------------
                # HTML REPORT
                # ---------------------------------------------------
                html_report = generate_html_report(
                    result
                )

                st.markdown(
                    download_button(
                        html_report,
                        "deepfake_report.html"
                    ),
                    unsafe_allow_html=True
                )

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("About")

st.sidebar.info("""
This AI system detects whether an uploaded audio file is real or fake using Deep Learning.
""")

st.sidebar.markdown("""
### Supported Formats
- WAV
- MP3
- OGG
- FLAC
- M4A
""")

st.sidebar.markdown("""
### Model Details
- TensorFlow CNN Model
- MFCC Features
- Deepfake Audio Detection
""")
