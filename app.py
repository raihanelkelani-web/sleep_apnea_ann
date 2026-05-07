import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------------
# LOAD MODEL + SCALER
# ----------------------------------
model = tf.keras.models.load_model("sleep_apnea_model.h5")
scaler = joblib.load("scaler.pkl")

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="Sleep Apnea AI", layout="wide")

# ----------------------------------
# TITLE STYLE (ORANGE HEADER)
# ----------------------------------
st.markdown("""
<style>
.main-title {
    background-color: #ff8c00;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SIDEBAR - ABOUT (ORANGE BOX)
# ----------------------------------
st.sidebar.markdown("""
<div style="
    background-color:#ff8c00;
    padding:12px;
    border-radius:10px;
    color:white;
    font-size:13px;
    line-height:1.4;
">
<b>ℹ About System</b><br><br>
AI-based sleep apnea detection using vital signs.<br>
Analyzes SpO₂, heart rate, breathing, snoring, and BMI.<br><br>
🔗 Supports Arduino-based monitoring device for continuous data collection.
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# SIDEBAR - PATIENT PROFILE + DEVICE + TIME
# ----------------------------------
st.sidebar.header("👤 Patient Profile")

now = datetime.now()
st.sidebar.write("🕒 Date & Time:", now.strftime("%Y-%m-%d %H:%M:%S"))

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 100, 30)
patient_id = st.sidebar.text_input("ID")

st.sidebar.header("🔌 Device Connection")

if st.sidebar.button("🔗 Connect Device"):
    st.sidebar.success("Device Connected Successfully")
    st.sidebar.info("Receiving data from Sleep Apnea Monitor...")

if st.sidebar.button("📥 Download 8–9 Hour Report"):
    st.sidebar.success("Report Generated")

    fake_report = f"""
Sleep Apnea Monitoring Report
--------------------------------
Patient: {name}
ID: {patient_id}
Age: {age}
Date: {now.strftime("%Y-%m-%d %H:%M:%S")}

Simulated 8–9 Hour Data:
- Average SpO2: --
- Average Heart Rate: --
- Breathing Rate: --
- Risk Score: --

Note: Simulated Arduino device report.
"""

    st.sidebar.download_button(
        "⬇ Download Report",
        fake_report,
        file_name="device_report.txt"
    )

st.sidebar.header("🚨 Alerts")

# ----------------------------------
# TITLE (ORANGE HEADER)
# ----------------------------------
st.markdown('<div class="main-title">💤 Sleep Apnea AI Monitoring System</div>', unsafe_allow_html=True)

# ----------------------------------
# INPUTS
# ----------------------------------
st.header("📋 Patient Input")

col1, col2, col3 = st.columns(3)

with col1:
    spo2 = st.number_input("SpO₂ (%)", 80, 100, 95)
    breathing_rate = st.number_input("Breathing Rate", 10, 30, 16)

with col2:
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 75)
    snoring = st.slider("Snoring Level", 0.0, 1.0, 0.3)

with col3:
    bmi = st.number_input("BMI", 10, 50, 25)

# ----------------------------------
# BUTTON
# ----------------------------------
if st.button("🚀 Generate Full Sleep Report"):

    # ---------------------------
    # AI PREDICTION
    # ---------------------------
    input_data = np.array([[spo2, heart_rate, breathing_rate, snoring, bmi]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    # ---------------------------
    # API CALCULATION
    # ---------------------------
    api = (
        (100 - spo2) * 0.4 +
        (heart_rate - 70) * 0.2 +
        (breathing_rate - 16) * 0.2 +
        (snoring * 30) +
        (bmi - 25) * 0.2
    )
    api = max(0, api)

    # ----------------------------------
    # VITAL SIGNS
    # ----------------------------------
    st.header("📊 Vital Signs")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f'<div style="padding:15px;background:#1abc9c;color:white;border-radius:10px">🫁 SpO₂<br>{spo2}%</div>', unsafe_allow_html=True)
    c2.markdown(f'<div style="padding:15px;background:#3498db;color:white;border-radius:10px">❤️ HR<br>{heart_rate}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div style="padding:15px;background:#9b59b6;color:white;border-radius:10px">🌬 Breath<br>{breathing_rate}</div>', unsafe_allow_html=True)
    c4.markdown(f'<div style="padding:15px;background:#e67e22;color:white;border-radius:10px">😴 Snore<br>{snoring}</div>', unsafe_allow_html=True)
    c5.markdown(f'<div style="padding:15px;background:#e74c3c;color:white;border-radius:10px">⚖ BMI<br>{bmi}</div>', unsafe_allow_html=True)

    # ----------------------------------
    # AI PREDICTION
    # ----------------------------------
    st.header("🧠 AI Risk Prediction")

    st.info("""
    🧠 Risk Scale:
    🟢 Low: 0.0 – 0.4  
    🟠 Moderate: 0.4 – 0.7  
    🔴 High: 0.7 – 1.0
    """)

    if prediction > 0.7:
        st.error(f"🔴 CRITICAL ({prediction:.2f})")
        st.sidebar.error("CRITICAL RISK")
    elif prediction > 0.4:
        st.warning(f"🟠 WARNING ({prediction:.2f})")
        st.sidebar.warning("MODERATE RISK")
    else:
        st.success(f"🟢 STABLE ({prediction:.2f})")
        st.sidebar.success("LOW RISK")

    # ----------------------------------
    # API DISPLAY
    # ----------------------------------
    st.header("📊 API (Apnea Performance Index)")

    st.metric("API Score", f"{api:.2f}")

    if api > 50:
        st.error("🔴 Severe Apnea Activity")
    elif api > 25:
        st.warning("🟠 Moderate Apnea Activity")
    else:
        st.success("🟢 Normal Breathing Pattern")

    # ----------------------------------
    # CLINICAL ANALYSIS
    # ----------------------------------
    st.header("🔍 Clinical Analysis")

    if spo2 < 92:
        st.write("Low oxygen level")
    if heart_rate > 90:
        st.write("High heart rate")
    if breathing_rate > 20:
        st.write("Rapid breathing")
    if snoring > 0.7:
        st.write("High snoring")
    if bmi > 30:
        st.write("High BMI")

    # ----------------------------------
    # SIMULATED GRAPHS
    # ----------------------------------
    st.header("📈 Simulated Sleep Signals")

    time = np.arange(60)

    spo2_signal = spo2 + np.random.normal(0, 0.8, 60)
    for i in range(10, 60, 15):
        spo2_signal[i:i+3] -= np.random.uniform(2, 5)

    breathing_signal = breathing_rate + np.random.normal(0, 1.5, 60)

    colg1, colg2 = st.columns(2)

    with colg1:
        st.subheader("🫁 SpO₂ Over Time")
        fig1, ax1 = plt.subplots()
        ax1.plot(time, spo2_signal)
        ax1.set_ylabel("SpO₂ (%)")
        ax1.set_xlabel("Time")
        st.pyplot(fig1)

    with colg2:
        st.subheader("🌬 Breathing Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(time, breathing_signal)
        ax2.set_ylabel("Breaths/min")
        ax2.set_xlabel("Time")
        st.pyplot(fig2)

    st.caption("We simulate physiological variability over time based on baseline readings to approximate real monitoring conditions.")

    # ----------------------------------
    # REPORT
    # ----------------------------------
    report = f"""
Sleep Apnea Report
-----------------------
Patient: {name}
Age: {age}
ID: {patient_id}
Date: {now.strftime("%Y-%m-%d %H:%M:%S")}

SpO₂: {spo2}
Heart Rate: {heart_rate}
Breathing Rate: {breathing_rate}
Snoring: {snoring}
BMI: {bmi}

AI Risk: {prediction:.2f}
API Score: {api:.2f}
"""

    st.session_state.report_data = report

# ----------------------------------
# DOWNLOAD BUTTON (FIXED)
# ----------------------------------
if "report_data" in st.session_state:
    st.download_button(
        "📄 Download Full Report",
        st.session_state.report_data,
        file_name="sleep_report.txt"
    )