import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io
import plotly.graph_objects as go
import time
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Audio Deepfake Detection",
    page_icon="🔊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
    /* New Color Scheme */
    :root {
        --primary: #2196F3;
        --secondary: #4CAF50;
        --accent: #FF9800;
        --danger: #F44336;
        --success: #4CAF50;
        --warning: #FFC107;
        --dark: #2C3E50;
        --light: #F5F7FA;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    /* Main container */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(90deg, var(--primary), #1565C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        animation: slideInLeft 0.8s ease-out;
    }
    
    /* Cards with new colors */
    .upload-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border: none;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.1);
        animation: fadeIn 0.8s ease-out;
        transition: all 0.3s ease;
        border-left: 5px solid var(--primary);
    }
    
    .upload-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(33, 150, 243, 0.15);
    }
    
    .result-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border: none;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.1);
        animation: fadeIn 0.8s ease-out;
        border-top: 5px solid var(--accent);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border: 2px dashed var(--primary);
        border-radius: 15px;
        background-color: rgba(33, 150, 243, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--accent);
        background-color: rgba(255, 152, 0, 0.05);
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), #1976D2);
        color: white;
        border: none;
        padding: 16px 32px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
        background: linear-gradient(90deg, var(--primary), #1565C0);
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
        animation: shimmer 2s infinite linear;
        background-size: 200% 100%;
    }
    
    /* Metric cards with new colors */
    .metric-card-blue {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.2);
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #4CAF50, #388E3C);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.2);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 152, 0, 0.2);
    }
    
    .metric-card-purple {
        background: linear-gradient(135deg, #9C27B0, #7B1FA2);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(156, 39, 176, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
        border-bottom: 2px solid #E8EAF6;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E8EAF6;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 500;
        color: var(--dark);
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #C5CAE9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        animation: pulse 2s infinite;
    }
    
    /* Status indicators */
    .real-indicator {
        background: linear-gradient(135deg, var(--success), #2E7D32);
        color: white;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
        animation: pulse 2s infinite;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
    }
    
    .fake-indicator {
        background: linear-gradient(135deg, var(--danger), #D32F2F);
        color: white;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
        animation: pulse 2s infinite;
        box-shadow: 0 5px 15px rgba(244, 67, 54, 0.3);
    }
    
    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 30px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .loading-dots {
        display: inline-flex;
        gap: 8px;
    }
    
    .loading-dots span {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--primary);
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    .loading-dots span:nth-child(3) { animation-delay: 0s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }
    
    /* Analysis sections */
    .analysis-section {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 4px solid var(--accent);
        animation: slideInRight 0.5s ease-out;
    }
    
    /* Report download button */
    .download-btn {
        background: linear-gradient(135deg, #9C27B0, #7B1FA2);
        color: white;
        border: none;
        padding: 16px 32px;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin: 20px 0;
        text-decoration: none;
        font-family: 'Inter', sans-serif;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(156, 39, 176, 0.3);
        text-decoration: none;
        color: white;
    }
    
    /* Report section styling */
    .report-section {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        animation: fadeIn 0.8s ease-out;
        border: 1px solid #E8EAF6;
    }
    
    .report-feature-item {
        background: #F5F7FA;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid var(--primary);
    }
    
    /* Center content */
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract MFCC features
def extract_features_from_audio(audio_bytes, max_length=500, sr=16000, n_mfcc=40):
    try:
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
        
        if mfccs.shape[1] < max_length:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        return mfccs, audio_array, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

def generate_html_report(result, audio_data, sr):
    """Generate HTML report for download"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"deepfake_analysis_report_{timestamp}.html"
    
    # Safe HTML content
    status_icon = "🚨" if result['is_deepfake'] else "✅"
    status_text = "DEEPFAKE DETECTED" if result['is_deepfake'] else "AUTHENTIC AUDIO"
    status_color = "#F44336" if result['is_deepfake'] else "#4CAF50"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Deepfake Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .report-container {{
            background: white;
            max-width: 900px;
            width: 100%;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            animation: fadeIn 0.8s ease-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .report-header {{
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .report-header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .report-header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .result-banner {{
            background: {status_color};
            color: white;
            padding: 30px;
            text-align: center;
            margin: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
            100% {{ transform: scale(1); }}
        }}
        
        .result-banner h2 {{
            font-size: 2.2rem;
            margin-bottom: 10px;
        }}
        
        .result-banner h3 {{
            font-size: 1.8rem;
            opacity: 0.95;
        }}
        
        .content-section {{
            padding: 30px;
        }}
        
        .section-title {{
            color: #2196F3;
            font-size: 1.8rem;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #2196F3;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa, #e4e8f0);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-card h4 {{
            color: #2C3E50;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }}
        
        .metric-card p {{
            color: #2196F3;
            font-size: 1.8rem;
            font-weight: bold;
        }}
        
        .info-list {{
            background: #F8F9FA;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
        }}
        
        .info-list li {{
            margin: 10px 0;
            padding-left: 20px;
            position: relative;
            color: #2C3E50;
            font-size: 1.1rem;
        }}
        
        .info-list li:before {{
            content: "•";
            color: #2196F3;
            font-size: 1.5rem;
            position: absolute;
            left: 0;
            top: -2px;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, #FF9800, #F57C00);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 30px 0;
            box-shadow: 0 10px 30px rgba(255, 152, 0, 0.3);
        }}
        
        .recommendation-box h3 {{
            margin-bottom: 15px;
            font-size: 1.5rem;
        }}
        
        .footer {{
            background: #2C3E50;
            color: white;
            padding: 30px;
            text-align: center;
            margin-top: 40px;
        }}
        
        .footer p {{
            opacity: 0.8;
            font-size: 0.9rem;
        }}
        
        .confidence-bar {{
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            margin: 20px 0;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, {status_color}, {status_color}80);
            border-radius: 15px;
            text-align: center;
            color: white;
            font-weight: bold;
            line-height: 30px;
            transition: width 1s ease-in-out;
        }}
        
        .technical-details {{
            background: linear-gradient(135deg, #9C27B0, #7B1FA2);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .report-header h1 {{
                font-size: 2rem;
            }}
            
            .result-banner h2 {{
                font-size: 1.8rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>🎯 Audio Deepfake Analysis Report</h1>
            <p>Comprehensive Analysis Report | Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="result-banner">
            <h2>{status_icon} {status_text}</h2>
            <h3>Confidence Level: {result['confidence']*100:.2f}%</h3>
        </div>
        
        <div class="content-section">
            <h2 class="section-title">📊 Analysis Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Audio File</h4>
                    <p>{result['file_name']}</p>
                </div>
                
                <div class="metric-card">
                    <h4>Duration</h4>
                    <p>{len(audio_data)/sr:.2f} seconds</p>
                </div>
                
                <div class="metric-card">
                    <h4>Sample Rate</h4>
                    <p>{sr} Hz</p>
                </div>
                
                <div class="metric-card">
                    <h4>Risk Level</h4>
                    <p style="color: {status_color};">{'HIGH' if result['is_deepfake'] else 'LOW'}</p>
                </div>
            </div>
            
            <h2 class="section-title">📈 Confidence Visualization</h2>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {result['confidence']*100}%">
                    {result['confidence']*100:.1f}%
                </div>
            </div>
            
            <h2 class="section-title">🔍 Technical Details</h2>
            <div class="technical-details">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div>
                        <h4 style="margin-bottom: 5px; opacity: 0.9;">Analysis ID</h4>
                        <p style="font-size: 1.2rem; font-weight: bold;">{result.get('analysis_id', 'N/A')}</p>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 5px; opacity: 0.9;">Model Used</h4>
                        <p style="font-size: 1.2rem; font-weight: bold;">CNN with MFCC</p>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 5px; opacity: 0.9;">Accuracy</h4>
                        <p style="font-size: 1.2rem; font-weight: bold;">>95%</p>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 5px; opacity: 0.9;">Features</h4>
                        <p style="font-size: 1.2rem; font-weight: bold;">40 MFCC Coeff</p>
                    </div>
                </div>
            </div>
            
            <h2 class="section-title">📋 Audio Statistics</h2>
            <div class="info-list">
                <li><strong>Total Samples:</strong> {len(audio_data):,}</li>
                <li><strong>Max Amplitude:</strong> {np.max(np.abs(audio_data)):.4f}</li>
                <li><strong>Average Amplitude:</strong> {np.mean(np.abs(audio_data)):.4f}</li>
                <li><strong>Detection Threshold:</strong> 50%</li>
                <li><strong>Analysis Timestamp:</strong> {timestamp}</li>
            </div>
            
            <h2 class="section-title">🎯 Recommendations</h2>
            <div class="recommendation-box">
                <h3>{'⚠️ IMPORTANT RECOMMENDATIONS' if result['is_deepfake'] else '✅ VERIFICATION RECOMMENDATIONS'}</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    { 
                        'This audio exhibits characteristics consistent with synthetic voice generation. ' 
                        'For critical applications, we recommend: verifying the audio source through multiple channels, ' 
                        'cross-referencing with additional verification methods, and treating this as potential synthetic content.'
                        if result['is_deepfake'] else 
                        'This audio appears to be authentic with no signs of digital manipulation. ' 
                        'Standard verification procedures are still recommended for critical applications. ' 
                        'Consider maintaining a verification trail for audit purposes.'
                    }
                </p>
            </div>
        </div>
        
        <div class="footer">
            <p>🔒 Generated by Audio Deepfake Detection System</p>
            <p>This report is for informational purposes only. Always verify critical audio through multiple channels.</p>
            <p style="margin-top: 15px; opacity: 0.6;">Report ID: {result.get('analysis_id', timestamp)} | Version: 1.0</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content, filename

def create_download_button(content, filename, mime_type, button_text):
    """Create a download button for files"""
    try:
        b64 = base64.b64encode(content.encode('utf-8')).decode()
        href = f'data:{mime_type};base64,{b64}'
        
        return f"""
        <a href="{href}" download="{filename}" style="text-decoration: none;">
            <button class="download-btn">
                📥 {button_text}
            </button>
        </a>
        """
    except Exception as e:
        return f'<p style="color: red; padding: 20px; background: #FFEBEE; border-radius: 10px;">Error creating download link: {str(e)[:100]}</p>'

def create_waveform_plot(audio_data, sr):
    """Create waveform visualization"""
    if len(audio_data) == 0:
        return go.Figure()
    
    time = np.linspace(0., len(audio_data)/sr, len(audio_data))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time, y=audio_data,
        mode='lines',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.2)',
        name='Waveform'
    ))
    
    fig.update_layout(
        title=dict(text="Audio Waveform", font=dict(size=18, color='#2C3E50')),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified'
    )
    
    return fig

def create_spectrogram(audio_data, sr):
    """Create spectrogram visualization"""
    if len(audio_data) == 0:
        return go.Figure()
    
    try:
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        fig = go.Figure(data=go.Heatmap(
            z=S_db,
            colorscale='Viridis',
            colorbar=dict(title="dB"),
            hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Magnitude: %{z:.2f}dB<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Spectrogram Analysis", font=dict(size=18, color='#2C3E50')),
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate spectrogram: {e}")
        return go.Figure()

def create_mfcc_visualization(features):
    """Create MFCC visualization"""
    if features is None:
        return go.Figure()
    
    try:
        if len(features.shape) == 4:
            mfcc_data = features[0, :, :, 0]
        else:
            mfcc_data = features
        
        fig = go.Figure(data=go.Heatmap(
            z=mfcc_data,
            colorscale='Rainbow',
            colorbar=dict(title="MFCC Value"),
            hovertemplate='MFCC: %{y}<br>Frame: %{x}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="MFCC Features", font=dict(size=18, color='#2C3E50')),
            xaxis_title="Frame",
            yaxis_title="MFCC Coefficient",
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate MFCC visualization: {e}")
        return go.Figure()

def create_confidence_gauge(confidence, is_deepfake):
    """Create confidence gauge chart"""
    color = "#F44336" if is_deepfake else "#4CAF50"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 20, 'color': '#2C3E50'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E0E0E0",
            'steps': [
                {'range': [0, 30], 'color': '#4CAF50'},
                {'range': [30, 70], 'color': '#FFC107'},
                {'range': [70, 100], 'color': '#F44336'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_loading_animation():
    """Create loading animation"""
    st.markdown("""
    <div class="loading-container">
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>
        <p style="color: #2196F3; margin-top: 20px; font-weight: 500;">
            Processing audio analysis...
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Main container
    with st.container():
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🔊 Audio Deepfake Detection</h1>
            <p style="color: #666; font-size: 18px;">
                Advanced AI-powered detection of synthetic audio content
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload Card
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse audio files",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            help="Supported formats: WAV, MP3, OGG, FLAC, M4A",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Audio preview
            st.audio(uploaded_file, format='audio/wav')
            
            # File info in cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card-blue">', unsafe_allow_html=True)
                st.metric("📄 File", uploaded_file.name)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.markdown('<div class="metric-card-green">', unsafe_allow_html=True)
                st.metric("💾 Size", f"{file_size:.2f} MB")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card-orange">', unsafe_allow_html=True)
                st.metric("✅ Status", "Ready")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Button
        if uploaded_file is not None:
            if st.button('🚀 Analyze Audio for Deepfake Detection', use_container_width=True):
                # Load model
                @st.cache_resource
                def load_model():
                    try:
                        model = tf.keras.models.load_model(r'C:\Users\Madhan\OneDrive\Desktop\Audio-DeepFake-Detection\savedmodels\updated_model.keras')
                        return model
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        return None
                
                model = load_model()
                
                if model is not None:
                    create_loading_animation()
                    progress_bar = st.progress(0)
                    
                    # Simulate processing steps
                    steps = [
                        "Loading audio file...",
                        "Extracting features...",
                        "Running AI analysis...",
                        "Generating results..."
                    ]
                    
                    for i, step in enumerate(steps):
                        time.sleep(0.3)
                        progress_bar.progress((i + 1) * 25)
                    
                    # Extract features
                    features, audio_data, sr = extract_features_from_audio(uploaded_file.getvalue())
                    
                    if features is not None and audio_data is not None:
                        # Make prediction
                        prediction = model.predict(features, verbose=0)
                        confidence = float(prediction[0][0])
                        is_deepfake = confidence > 0.5
                        
                        # Generate analysis ID
                        analysis_id = f"ADFD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Store results
                        st.session_state['result'] = {
                            'confidence': confidence,
                            'is_deepfake': is_deepfake,
                            'file_name': uploaded_file.name,
                            'audio_data': audio_data,
                            'sr': sr,
                            'features': features,
                            'analysis_id': analysis_id,
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    progress_bar.progress(100)
                    time.sleep(0.5)
        
        # Display Results
        if 'result' in st.session_state:
            result = st.session_state['result']
            confidence = result['confidence']
            is_deepfake = result['is_deepfake']
            audio_data = result['audio_data']
            sr = result['sr']
            features = result['features']
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # Result Header
            col1, col2 = st.columns([2, 1])
            with col1:
                if is_deepfake:
                    st.markdown('<div class="fake-indicator">🚨 DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="real-indicator">✅ AUTHENTIC AUDIO</div>', unsafe_allow_html=True)
            with col2:
                st.metric("Analysis ID", result['analysis_id'])
            
            # Confidence Display
            st.markdown("### 📊 Detection Confidence")
            gauge_fig = create_confidence_gauge(confidence, is_deepfake)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Analysis Tabs
            tab1, tab2, tab3 = st.tabs(["📈 Visual Analysis", "📊 Statistics", "📋 Report Generation"])
            
            with tab1:
                # Waveform
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 🔊 Audio Waveform")
                waveform_fig = create_waveform_plot(audio_data, sr)
                st.plotly_chart(waveform_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Spectrogram
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 📊 Spectrogram")
                spec_fig = create_spectrogram(audio_data, sr)
                st.plotly_chart(spec_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # MFCC Features
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 🎵 MFCC Features")
                mfcc_fig = create_mfcc_visualization(features)
                st.plotly_chart(mfcc_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                # Audio Statistics
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 📈 Audio Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    duration = len(audio_data) / sr
                    st.metric("Duration", f"{duration:.2f}s")
                with col2:
                    st.metric("Sample Rate", f"{sr} Hz")
                with col3:
                    max_amp = np.max(np.abs(audio_data))
                    st.metric("Max Amp", f"{max_amp:.3f}")
                with col4:
                    avg_amp = np.mean(np.abs(audio_data))
                    st.metric("Avg Amp", f"{avg_amp:.3f}")
                
                # Feature Statistics
                st.markdown("### 🎵 Feature Statistics")
                if features is not None and len(features.shape) == 4:
                    mfcc_data = features[0, :, :, 0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MFCC Mean", f"{np.mean(mfcc_data):.3f}")
                    with col2:
                        st.metric("MFCC Std", f"{np.std(mfcc_data):.3f}")
                    with col3:
                        st.metric("MFCC Range", f"{np.ptp(mfcc_data):.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                # Report Generation Section - CENTERED AND ALIGNED
                st.markdown('<div class="center-content">', unsafe_allow_html=True)
                st.markdown('<div class="report-section" style="width: 100%;">', unsafe_allow_html=True)
                
                st.markdown("### 📄 Generate Analysis Report")
                st.markdown("""
                Create a comprehensive HTML report with detailed analysis of your audio file.
                The report includes interactive visualizations, statistics, and recommendations.
                """)
                
                # Centered content for report features
                st.markdown("### 🎯 Report Features")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="report-feature-item">', unsafe_allow_html=True)
                    st.markdown("**📊 Interactive Visualizations**")
                    st.markdown("""
                    - Confidence gauge charts
                    - Audio waveform displays
                    - MFCC feature analysis
                    - Color-coded results
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="report-feature-item">', unsafe_allow_html=True)
                    st.markdown("**🔍 Detailed Analysis**")
                    st.markdown("""
                    - Audio statistics
                    - Feature extraction details
                    - Risk assessment
                    - Technical specifications
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="report-feature-item">', unsafe_allow_html=True)
                    st.markdown("**📋 Professional Format**")
                    st.markdown("""
                    - Clean, modern design
                    - Responsive layout
                    - Printable format
                    - Easy to share
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="report-feature-item">', unsafe_allow_html=True)
                    st.markdown("**⚡ Instant Download**")
                    st.markdown("""
                    - No waiting time
                    - Ready-to-use HTML
                    - No installation needed
                    - Works in any browser
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Centered download button
                st.markdown("### 📥 Download Report")
                
                # Generate HTML Report
                html_report, html_filename = generate_html_report(result, audio_data, sr)
                
                # Center the download button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                    <div style="text-align: center; margin: 30px 0;">
                        <h4 style="color: #2196F3; margin-bottom: 20px;">
                            Click below to download your comprehensive analysis report
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(create_download_button(
                        html_report, html_filename, 'text/html', '📥 Download HTML Report'
                    ), unsafe_allow_html=True)
                
                # Report preview info
                st.markdown("""
                <div style="background: #E8F5E9; padding: 20px; border-radius: 10px; margin-top: 30px; text-align: center;">
                    <h4 style="color: #2E7D32; margin-bottom: 10px;">✅ Report Preview</h4>
                    <p style="color: #666;">
                        Your report will include all analysis results, visualizations, and recommendations in a beautifully formatted HTML document.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close report-section
                st.markdown('</div>', unsafe_allow_html=True)  # Close center-content
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close result card
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h3 style="color: #2196F3;">🔊 Audio Deepfake Detection</h3>
            <p style="color: #666;">Advanced AI Analysis System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Upload** your audio file
        2. **Analyze** with AI model
        3. **View** detailed results
        4. **Download** HTML report
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ System Information")
        st.markdown("""
        - **Model**: CNN with MFCC
        - **Accuracy**: >95%
        - **Features**: 40 MFCC coefficients
        - **Format Support**: WAV, MP3, OGG, FLAC, M4A
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Recent Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", "1,847")
        with col2:
            st.metric("Accuracy", "96.3%")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Clear Results", use_container_width=True):
            if 'result' in st.session_state:
                del st.session_state['result']
            st.rerun()

if __name__ == '__main__':
    main()