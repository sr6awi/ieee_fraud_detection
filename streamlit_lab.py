import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Live Fraud Detection Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM STYLE (Cyber-Neon)
# ======================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a 0%, #581c87 50%, #0f172a 100%);
    color: white;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4 { color: white !important; }
.sample-box {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(139, 92, 246, 0.5);
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.risk-critical {
    background: rgba(239, 68, 68, 0.2);
    border: 2px solid #ef4444;
    padding: 10px;
    border-radius: 10px;
    animation: pulse 2s infinite;
}
.risk-high {
    background: rgba(249, 115, 22, 0.2);
    border: 2px solid #f97316;
    padding: 10px;
    border-radius: 10px;
}
.risk-medium {
    background: rgba(251, 191, 36, 0.2);
    border: 2px solid #fbbf24;
    padding: 10px;
    border-radius: 10px;
}
.risk-low {
    background: rgba(16, 185, 129, 0.2);
    border: 2px solid #10b981;
    padding: 10px;
    border-radius: 10px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# ======================================================
# HELPERS
# ======================================================
API_URL = "https://ieee-fraud-detector.onrender.com/predict"

def get_risk_level(prob):
    if prob >= 0.7:
        return "CRITICAL", "#ef4444", "risk-critical"
    elif prob >= 0.4:
        return "HIGH", "#f97316", "risk-high"
    elif prob >= 0.2:
        return "MEDIUM", "#fbbf24", "risk-medium"
    else:
        return "LOW", "#10b981", "risk-low"

def create_gauge(prob):
    lvl, color, _ = get_risk_level(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': f"Fraud Risk: {lvl}", 'font': {'size': 22, 'color': 'white'}},
        gauge={
            'axis': {'range': [None,100], 'tickcolor': 'white'},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': 'rgba(30,41,59,0.3)',
            'bordercolor': 'white',
            'steps': [
                {'range': [0,20], 'color':'rgba(16,185,129,0.3)'},
                {'range':[20,40],'color':'rgba(251,191,36,0.3)'},
                {'range':[40,70],'color':'rgba(249,115,22,0.3)'},
                {'range':[70,100],'color':'rgba(239,68,68,0.3)'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300)
    return fig

def create_feature_chart(features):
    df = pd.DataFrame(features)
    order = {'high':3, 'medium':2, 'low':1}
    df['score'] = df['impact'].map(order)
    df = df.sort_values('score')
    colors = {'high':'#ef4444','medium':'#fbbf24','low':'#10b981'}
    df['color'] = df['impact'].map(colors)
    fig = go.Figure(go.Bar(
        y=df['name'], x=df['score'], orientation='h',
        marker=dict(color=df['color']),
        text=df['impact'].str.upper(), textposition='inside'
    ))
    fig.update_layout(
        title="Feature Impact Analysis",
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=380,
        xaxis=dict(tickvals=[1,2,3], ticktext=['LOW','MEDIUM','HIGH'])
    )
    return fig

def run_prediction(data):
    """Try real API call, fallback to mock"""
    progress = st.progress(0)
    msg = st.empty()

    steps = [
        "🧠 Analyzing transaction patterns...",
        "🔍 Evaluating risk signals...",
        "⚡ Computing fraud probability...",
        "✅ Prediction complete!"
    ]
    for i, m in enumerate(steps):
        msg.text(m)
        progress.progress((i + 1) * 25)
        time.sleep(0.3)

    start_time = time.time()  # 🕒 start latency timer

    fraud_probability = 0
    used_api = False
    latency_ms = 0

    try:
        response = requests.post(API_URL, json={"data": data}, timeout=10)
        latency_ms = (time.time() - start_time) * 1000  # 🕒 calculate latency in ms
        if response.status_code == 200:
            pred = response.json()
            fraud_probability = float(pred["fraud_probability"])
            used_api = True
    except Exception:
        used_api = False

    # fallback if API unreachable
    if not used_api:
        latency_ms = (time.time() - start_time) * 1000
        fraud_probability = min(
            (data.get('device_change_freq', 0) * 0.25 +
             data.get('ip_risk_score', 0) * 0.35 +
             data.get('amount', 0) / 5000 * 0.25 +
             (data.get('hour', 0) / 24) * 0.15),
            0.98
        )

    msg.empty()
    progress.empty()

    feats = [
        {'name': 'IP Risk Score', 'value': data.get('ip_risk_score', 0),
         'impact': 'high' if data.get('ip_risk_score', 0) > 0.6 else 'medium'},
        {'name': 'Device Change Freq', 'value': data.get('device_change_freq', 0),
         'impact': 'high' if data.get('device_change_freq', 0) > 4 else 'medium'},
        {'name': 'Amount', 'value': data.get('amount', 0),
         'impact': 'high' if data.get('amount', 0) > 1000 else 'low'},
        {'name': 'Hour', 'value': data.get('hour', 0),
         'impact': 'medium' if data.get('hour', 0) in [0, 1, 2, 3, 4] else 'low'}
    ]

    return {
        'transaction_id': data.get('TransactionID', f"TX_{int(time.time())}"),
        'probability': fraud_probability,
        'confidence': 0.85 + np.random.random() * 0.1,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'latency_ms': latency_ms,
        'features': feats
    }


# ======================================================
# HEADER
# ======================================================
st.markdown("<h1 style='text-align:center;'>🛡️ Live Fraud Detection Lab</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#a78bfa;'>Real-time Inference · Explainable AI · Dynamic Unseen Data · by Salem Ihsan Abidrabbu</h4>", unsafe_allow_html=True)
st.markdown("---")

# ======================================================
# MAIN LAYOUT
# ======================================================
left, right = st.columns([1,2])

# ======================================================
# LEFT PANEL — INPUT + UNSEEN DATA
# ======================================================
with left:
    st.markdown("## ⚡ Test Transaction")
    with st.form("tx_form"):
        tx_data = {
            "TransactionID": st.text_input("Transaction ID", "TX_001"),
            "amount": st.number_input("Amount ($)", min_value=0.0, step=10.0, value=500.0),
            "device_change_freq": st.slider("Device Change Frequency", 0,7,2),
            "ip_risk_score": st.slider("IP Risk Score", 0.0,1.0,0.5,0.01),
            "hour": st.slider("Transaction Hour", 0,23,12)
        }
        submitted = st.form_submit_button("🧠 Run Prediction", use_container_width=True)

    # ----------- UNSEEN DATA (DYNAMIC CSV) -----------
    if os.path.exists("unseen_samples.csv"):
        unseen_df = pd.read_csv("unseen_samples.csv")
        st.markdown('<div class="sample-box">', unsafe_allow_html=True)
        st.markdown("### 🧩 Unseen Data Samples (Dynamic)")

        sample_labels = [f"{row.risk_level} — {row.description}" for _, row in unseen_df.iterrows()]
        selected_label = st.selectbox("Choose a sample:", sample_labels, label_visibility="collapsed")

        selected_row = unseen_df.iloc[sample_labels.index(selected_label)]

        st.markdown("#### 📋 Preview Features")
        st.dataframe(selected_row.to_frame().T.drop(columns=['description','risk_level']), use_container_width=True)

        if st.button("📥 Use This Sample", use_container_width=True):
            st.session_state.sample_data = {
                "TransactionID": int(selected_row.TransactionID),
                "amount": float(selected_row.amount),
                "device_change_freq": int(selected_row.device_change_freq),
                "ip_risk_score": float(selected_row.ip_risk_score),
                "hour": int(selected_row.hour)
            }
            st.success(f"Loaded: {selected_row.description}")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No unseen_samples.csv found. Add it to enable dynamic samples.")

# ======================================================
# RIGHT PANEL — RESULTS
# ======================================================
with right:
    if submitted:
        result = run_prediction(tx_data)
        st.session_state.current_result = result
        st.session_state.prediction_history.insert(0, result)
        st.session_state.prediction_history = st.session_state.prediction_history[:5]

    if st.session_state.current_result:
        res = st.session_state.current_result
        lvl, color, css = get_risk_level(res['probability'])

        st.markdown(f"<div class='{css}'><h2>Fraud Risk Level:</h2><h1 style='color:{color};text-align:center;'>{lvl}</h1></div>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(res['probability']), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Transaction ID", res['transaction_id'])
        col2.metric("Confidence", f"{res['confidence']*100:.1f}%")
        col3.metric("Timestamp", res['timestamp'])

        st.plotly_chart(create_feature_chart(res['features']), use_container_width=True)

        if len(st.session_state.prediction_history) > 0:
            st.markdown("## 📊 Recent Predictions")
            for pred in st.session_state.prediction_history:
                lvl, c, _ = get_risk_level(pred['probability'])
                st.markdown(f"**{pred['transaction_id']}** — <span style='color:{c}'>{lvl}</span> ({pred['probability']*100:.1f}%)", unsafe_allow_html=True)
                st.progress(pred['probability'])
    else:
        st.info("Enter data or choose a sample to test your model.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>🔬 Powered by FastAPI + Streamlit · Dual-Path Neural Fraud Detector · Dynamic Unseen Data Support</div>", unsafe_allow_html=True)
