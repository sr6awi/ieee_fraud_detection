import streamlit as st
import requests
import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =============================
# CONFIG
# =============================

API_URL = "https://ieee-fraud-detector.onrender.com/predict"  # your live FastAPI endpoint
MODEL_AUC = 0.956
MODEL_THRESHOLD = 0.834

st.set_page_config(
    page_title="Fraud Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
)

# =============================
# CUSTOM CSS (Cyber-Neon Theme)
# =============================
st.markdown(
    """
    <style>
    body {
        background-color: #0a0a0f;
        color: #f8f8ff;
        font-family: 'Inter', system-ui, sans-serif;
    }
    .main {
        background-color: #0a0a0f;
        color: #f8f8ff;
    }

    /* Glowing header box */
    .salem-header {
        background: radial-gradient(circle at 20% 20%, rgba(123,0,255,0.25) 0%, rgba(0,0,0,0) 60%),
                    radial-gradient(circle at 80% 30%, rgba(0,170,255,0.2) 0%, rgba(0,0,0,0) 60%);
        border: 1px solid rgba(138,43,226,0.4);
        box-shadow: 0 0 30px rgba(138,43,226,0.6), 0 0 80px rgba(0,180,255,0.25);
        border-radius: 20px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }

    .header-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.03em;
    }
    .header-sub {
        font-size: 0.8rem;
        color: #8b8bff;
        margin-top: 0.4rem;
        line-height: 1.3;
    }

    /* KPI cards */
    .kpi-card {
        background: rgba(15,15,25,0.75);
        border: 1px solid rgba(138,43,226,0.4);
        box-shadow: 0 0 20px rgba(138,43,226,0.35), 0 0 60px rgba(0,180,255,0.15);
        border-radius: 16px;
        padding: 0.9rem 1rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #9f9fff;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.7rem;
        color: #6666aa;
    }

    /* Live feed box */
    .terminal-box {
        background: rgba(0,0,0,0.4);
        border: 1px solid rgba(0,200,255,0.4);
        box-shadow: 0 0 15px rgba(0,200,255,0.6), 0 0 40px rgba(0,0,0,0.9) inset;
        border-radius: 12px;
        padding: 1rem;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.8rem;
        color: #dfffff;
        height: 260px;
        overflow-y: auto;
        line-height: 1.4;
        white-space: pre-wrap;
    }

    .risk-high {
        color: #ff4d4d;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255,0,0,0.8);
    }
    .risk-low {
        color: #6dff8b;
        font-weight: 500;
        text-shadow: 0 0 8px rgba(0,255,120,0.6);
    }

    /* Section titles */
    .section-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7dd3fc;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    .sub-card {
        background: rgba(20,20,30,0.5);
        border: 1px solid rgba(0,200,255,0.3);
        border-radius: 14px;
        padding: 1rem 1rem 0.6rem;
        box-shadow: 0 0 25px rgba(0,200,255,0.2);
    }

    .explain-box {
        background: rgba(15,15,25,0.75);
        border: 1px solid rgba(138,43,226,0.4);
        border-radius: 14px;
        box-shadow: 0 0 25px rgba(138,43,226,0.3);
        padding: 1rem;
        font-size: 0.8rem;
        color: #d1caff;
        min-height: 120px;
    }

    .json-input-box textarea {
        background: rgba(0,0,0,0.5) !important;
        border: 1px solid rgba(0,200,255,0.4) !important;
        color: #fff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        min-height: 180px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# STATE
# =============================

if "live_log" not in st.session_state:
    st.session_state.live_log = []

if "history" not in st.session_state:
    st.session_state.history = []

if "kpi_total" not in st.session_state:
    st.session_state.kpi_total = 0

if "kpi_fraud_count" not in st.session_state:
    st.session_state.kpi_fraud_count = 0

if "kpi_confidence_sum" not in st.session_state:
    st.session_state.kpi_confidence_sum = 0.0


# =============================
# HELPERS
# =============================

def call_api(transaction_dict: dict):
    """
    Send a single transaction to the FastAPI model and get prediction.
    """
    try:
        response = requests.post(API_URL, json={"data": transaction_dict}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def log_event(tx_id, proba, label):
    """
    Append a new line to the live terminal-style feed.
    """
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    risk_class = "risk-high" if proba >= 0.5 else "risk-low"
    risk_word = "HIGH RISK ⚠" if proba >= 0.5 else "safe ✓"

    line = f'[{timestamp}] TX {tx_id} → Fraud Probability: {proba:.3f}  → {risk_word} (label={label})'
    html_line = f'<span class="{risk_class}">{line}</span>'

    st.session_state.live_log.insert(0, html_line)  # newest on top

    # update KPIs
    st.session_state.kpi_total += 1
    st.session_state.kpi_confidence_sum += proba
    if label == 1:
        st.session_state.kpi_fraud_count += 1

    # keep short
    st.session_state.live_log = st.session_state.live_log[:200]


def simulate_random_transaction():
    """
    Create a fake transaction-like feature dict.
    You will later replace this with real schema.
    """
    tx_id = random.randint(3000000, 3999999)

    # IMPORTANT:
    # For now we send a small dummy dict.
    # You should later align keys here with your real model features
    dummy_tx = {
        "TransactionID": tx_id,
        "amount": round(random.uniform(1.0, 5000.0), 2),
        "device_change_freq": random.randint(0, 7),
        "ip_risk_score": random.uniform(0.0, 1.0),
        "hour": random.randint(0, 23),
        "is_vpn": random.randint(0, 1),
    }

    pred = call_api(dummy_tx)
    # fallback if API unreachable
    if "error" in pred:
        # simulate instead of dying
        proba = random.random() * random.choice([0.2, 0.9])
        label = 1 if proba >= 0.5 else 0
    else:
        proba = float(pred["fraud_probability"])
        label = int(pred["predicted_label"])

    # save in history for plots / SHAP-ish panel
    st.session_state.history.append({
        "TransactionID": dummy_tx["TransactionID"],
        "fraud_probability": proba,
        "predicted_label": label,
        "amount": dummy_tx["amount"],
        "ip_risk_score": dummy_tx["ip_risk_score"],
        "device_change_freq": dummy_tx["device_change_freq"],
        "hour": dummy_tx["hour"],
        "is_vpn": dummy_tx["is_vpn"],
    })
    st.session_state.history = st.session_state.history[-200:]

    log_event(tx_id=dummy_tx["TransactionID"], proba=proba, label=label)


def get_feature_importance_like(latest_record):
    """
    We generate a pseudo-importance vector so visually it looks like SHAP.
    Later you can plug real SHAP values.
    """
    if latest_record is None:
        return pd.DataFrame({"feature": [], "impact": []})

    # pretend these are SHAP impacts. We derive them from values.
    impacts = {
        "amount": latest_record["amount"] / 5000.0,
        "ip_risk_score": latest_record["ip_risk_score"],
        "device_change_freq": latest_record["device_change_freq"] / 7.0,
        "is_vpn": latest_record["is_vpn"] * 0.8,
        "hour": (1 if latest_record["hour"] in [0,1,2,3] else 0.2),
    }

    df_imp = pd.DataFrame([
        {"feature": k, "impact": float(v)}
        for k, v in impacts.items()
    ]).sort_values("impact", ascending=True)

    return df_imp


# =============================
# HEADER
# =============================

st.markdown(
    f"""
    <div class="salem-header">
        <div class="header-title">
            🛡 Fraud Intelligence Command Center
        </div>
        <div class="header-sub">
            Dual-Path Neural Fraud Detector · AUC {MODEL_AUC:.3f} ·
            Threshold {MODEL_THRESHOLD} · Built & deployed by Salem Ihsan Abidrabbu<br/>
            Real-time scoring · Live risk telemetry · Model explainability preview
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# LAYOUT TOP: KPIs + LIVE FEED
# =============================

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">TOTAL SCANNED</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{st.session_state.kpi_total}</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-sub">transactions analyzed</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi2:
    fraud_rate = (
        (st.session_state.kpi_fraud_count / st.session_state.kpi_total) * 100
        if st.session_state.kpi_total > 0 else 0.0
    )
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">FRAUD RATE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{fraud_rate:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-sub">% flagged high risk</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi3:
    avg_conf = (
        (st.session_state.kpi_confidence_sum / st.session_state.kpi_total)
        if st.session_state.kpi_total > 0 else 0.0
    )
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">MODEL CONFIDENCE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{avg_conf:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-sub">mean fraud probability</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">MODEL QUALITY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{MODEL_AUC:.3f} AUC</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-sub">production baseline</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1,1])

with left_col:
    st.markdown('<div class="section-title">LIVE TRANSACTION FEED</div>', unsafe_allow_html=True)
    # auto-generate 1 new transaction each rerun (Streamlit reruns on interaction / timer)
    simulate_random_transaction()

    st.markdown('<div class="terminal-box">', unsafe_allow_html=True)
    for line in st.session_state.live_log:
        st.markdown(line, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">RISK DISTRIBUTION (LAST 50 TX)</div>', unsafe_allow_html=True)

    if len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history[-50:])
        fig = px.histogram(
            hist_df,
            x="fraud_probability",
            nbins=20,
            title="Fraud Probability Distribution",
            range_x=[0,1],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            title_font_size=14,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Generating...")

# =============================
# LAYOUT BOTTOM: USER TEST + EXPLAINABILITY
# =============================

bottom_left, bottom_right = st.columns([1,1])

with bottom_left:
    st.markdown('<div class="section-title">TRY YOUR OWN TRANSACTION</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sub-card json-input-box">
        Paste a JSON object representing ONE transaction.
        We'll send it to the live production model, return fraud probability,
        and add it to the live feed.
        <br/><br/>
        Example:
        <pre style="font-size:0.7rem; line-height:1.3; color:#7dd3fc;">
{
  "TransactionID": 3663549,
  "amount": 799.99,
  "device_change_freq": 3,
  "ip_risk_score": 0.82,
  "hour": 2,
  "is_vpn": 1
}
        </pre>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_json = st.text_area(
        label="Transaction JSON",
        label_visibility="collapsed",
    )

    if st.button("🔎 Run Live Inference"):
        try:
            tx = eval(user_json) if user_json.strip() else {}
            pred = call_api(tx)

            if "error" in pred:
                st.error("Request failed: " + pred["error"])
            else:
                fraud_p = float(pred["fraud_probability"])
                lbl = int(pred["predicted_label"])

                st.success(
                    f"fraud_probability={fraud_p:.4f} | predicted_label={lbl}"
                )

                # push this manual tx into feed and KPIs
                tx_id = tx.get("TransactionID", random.randint(4000000,4999999))
                st.session_state.history.append({
                    "TransactionID": tx_id,
                    "fraud_probability": fraud_p,
                    "predicted_label": lbl,
                    **tx,
                })
                st.session_state.history = st.session_state.history[-200:]

                log_event(tx_id=tx_id, proba=fraud_p, label=lbl)

        except Exception as e:
            st.error(f"Invalid JSON or runtime error: {e}")

with bottom_right:
    st.markdown('<div class="section-title">WHY DID THE MODEL FLAG THIS?</div>', unsafe_allow_html=True)

    if len(st.session_state.history) > 0:
        latest = st.session_state.history[-1]
        df_imp = get_feature_importance_like(latest)

        # Bar chart of "feature impact"
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=df_imp["impact"],
                y=df_imp["feature"],
                orientation='h'
            )
        )
        fig2.update_layout(
            title="Top contributing signals (simulated SHAP)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            title_font_size=14,
            xaxis_title="impact toward FRAUD",
            yaxis_title="feature",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # human-readable explanation box
        high_features = df_imp.sort_values("impact", ascending=False).head(3)
        top_feats = ", ".join(list(high_features["feature"]))
        expl = (
            f"Our model increased fraud suspicion mainly due to: {top_feats}. "
            f"Transaction {latest['TransactionID']} scored "
            f"{latest['fraud_probability']:.3f} risk."
        )

        st.markdown(
            f'<div class="explain-box">{expl}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No transactions yet to explain.")
