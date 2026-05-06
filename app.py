import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import io
import plotly.graph_objects as go
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Audio Deepfake Detection",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# MODERN UI CSS
# =========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Animated Background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #1e293b, #0f172a);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Header */

.main-header {
    text-align: center;
    padding: 40px 20px;
}

.main-header h1 {
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #cbd5e1;
    font-size: 1.2rem;
}

/* Cards */

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: 0.4s;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(56,189,248,0.25);
}

/* File uploader */

.stFileUploader > div > div {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #38bdf8;
    border-radius: 20px;
    padding: 25px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(135deg,#38bdf8,#6366f1,#ec4899);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 16px;
    width: 100%;
    font-weight: 600;
    font-size: 16px;
    transition: 0.4s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 30px rgba(236,72,153,0.4);
}

/* Tabs */

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    color: white;
    margin-right: 10px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#38bdf8,#6366f1);
}

/* Metric Cards */

.metric-card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    color: white;
    margin-bottom: 15px;
}

/* Result Indicators */

.real-indicator {
    background: linear-gradient(135deg,#22c55e,#15803d);
    color: white;
    padding: 15px 25px;
    border-radius: 40px;
    display: inline-block;
    font-weight: bold;
}

.fake-indicator {
    background: linear-gradient(135deg,#ef4444,#b91c1c);
    color: white;
    padding: 15px 25px;
    border-radius: 40px;
    display: inline-block;
    font-weight: bold;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Progress */

.stProgress > div > div > div {
    background: linear-gradient(90deg,#38bdf8,#6366f1,#ec4899);
}

/* Audio */

audio {
    width: 100%;
    margin-top: 15px;
}

/* Scrollbar */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(#38bdf8,#6366f1);
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("updated_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(audio_bytes):
    try:
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        mfccs = librosa.feature.mfcc(
            y=audio_array,
            sr=sr,
            n_mfcc=40
        )

        max_length = 500

        if mfccs.shape[1] < max_length:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(
                mfccs,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )
        else:
            mfccs = mfccs[:, :max_length]

        mfccs = mfccs.reshape(1, 40, 500, 1)

        return mfccs, audio_array, sr

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# =========================
# WAVEFORM
# =========================

def waveform_plot(audio_data, sr):

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
        fill='tozeroy'
    ))

    fig.update_layout(
        title="Audio Waveform",
        template="plotly_dark",
        height=350
    )

    return fig

# =========================
# MFCC PLOT
# =========================

def mfcc_plot(features):

    mfcc_data = features[0, :, :, 0]

    fig = go.Figure(data=go.Heatmap(
        z=mfcc_data,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title="MFCC Features",
        template="plotly_dark",
        height=350
    )

    return fig

# =========================
# HEADER
# =========================

st.markdown("""
<div class="main-header">
    <h1>🔊 Deepfake Audio Detection</h1>
    <p>AI Powered Synthetic Voice Detection System</p>
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN CARD
# =========================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("📤 Upload Audio File")

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=['wav', 'mp3', 'ogg', 'flac', 'm4a']
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h4>📄 File Name</h4>
        <p>{uploaded_file.name}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        st.markdown(f"""
        <div class="metric-card">
        <h4>💾 File Size</h4>
        <p>{size_mb:.2f} MB</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>✅ Status</h4>
        <p>Ready</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ANALYZE BUTTON
# =========================

if uploaded_file is not None:

    if st.button("🚀 Analyze Audio"):

        if model is not None:

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            features, audio_data, sr = extract_features(
                uploaded_file.getvalue()
            )

            if features is not None:

                prediction = model.predict(features)

                confidence = float(prediction[0][0])

                is_fake = confidence > 0.5

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)

                if is_fake:
                    st.markdown(
                        '<div class="fake-indicator">🚨 DEEPFAKE DETECTED</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="real-indicator">✅ AUTHENTIC AUDIO</div>',
                        unsafe_allow_html=True
                    )

                st.write("")

                st.metric(
                    "Confidence Score",
                    f"{confidence*100:.2f}%"
                )

                tabs = st.tabs([
                    "📈 Waveform",
                    "🎵 MFCC Features",
                    "📊 Statistics"
                ])

                with tabs[0]:
                    fig1 = waveform_plot(audio_data, sr)
                    st.plotly_chart(
                        fig1,
                        use_container_width=True
                    )

                with tabs[1]:
                    fig2 = mfcc_plot(features)
                    st.plotly_chart(
                        fig2,
                        use_container_width=True
                    )

                with tabs[2]:

                    duration = len(audio_data) / sr

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Duration",
                            f"{duration:.2f} sec"
                        )

                    with c2:
                        st.metric(
                            "Sample Rate",
                            f"{sr} Hz"
                        )

                    with c3:
                        st.metric(
                            "Analysis Time",
                            datetime.now().strftime("%H:%M:%S")
                        )

                st.markdown('</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🔊 Deepfake Detector")

    st.markdown("---")

    st.markdown("""
    ### 🎯 Features

    ✅ AI Detection  
    ✅ MFCC Analysis  
    ✅ Waveform Visualization  
    ✅ Interactive Dashboard  
    ✅ Modern UI  
    """)

    st.markdown("---")

    st.markdown("""
    ### ⚙️ Model Details

    - Model: CNN
    - Input: MFCC
    - Accuracy: 95%+
    - Audio Formats:
      WAV, MP3, OGG,
      FLAC, M4A
    """)

    st.markdown("---")

    st.info(
        "Upload an audio file and analyze whether it is authentic or AI generated."
    )
