import streamlit as st
import numpy as np
import pywt
from scipy.signal import welch, butter, filtfilt
import plotly.graph_objects as go
import websocket
import threading
import json
from sklearn.ensemble import RandomForestClassifier


# ------------------------------
# Signal Generation Functions
# ------------------------------

def generate_imbalance_signal(frequency, sampling_rate, duration, imbalance_amp=0.5):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = np.sin(2 * np.pi * frequency * t) + imbalance_amp * np.sin(2 * np.pi * 2 * frequency * t)
    return t, signal


def generate_misalignment_signal(frequency, mod_freq, mod_index, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    carrier = np.sin(2 * np.pi * frequency * t)
    modulator = 1 + mod_index * np.sin(2 * np.pi * mod_freq * t)
    signal = carrier * modulator
    return t, signal


def generate_bearing_fault_signal(bpfo, sampling_rate, duration, fault_amp=0.5):
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = fault_amp * np.sin(2 * np.pi * bpfo * t) + np.random.normal(0, 0.1, len(t))
    return t, signal


# ------------------------------
# DWT Analysis Function
# ------------------------------

def perform_dwt_analysis(signal, wavelet="db4", level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs


# ------------------------------
# Signal Filtering Function
# ------------------------------

def butter_filter(signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


# ------------------------------
# WebSocket Integration for Real-Time Data
# ------------------------------

devices = ["Machine 1", "Machine 2", "Machine 3"]
device_data = {device: [] for device in devices}


def on_message(ws, message):
    data = json.loads(message)
    device_id = data["device_id"]
    device_data[device_id].append([data["x"], data["y"], data["z"]])

    # Visualize data for the specific device
    if len(device_data[device_id]) >= 160:
        # Process the real-time data here (you can add a custom function for processing)
        pass


def websocket_thread():
    ws = websocket.WebSocketApp("ws://192.168.137.114/ws", on_message=on_message)
    ws.run_forever()


if "real_time_signal" not in st.session_state:
    st.session_state["real_time_signal"] = np.zeros(100)

thread = threading.Thread(target=websocket_thread)
thread.daemon = True
thread.start()


# ------------------------------
# Fault Classification and Severity Calculation
# ------------------------------

def extract_features_from_dwt(coeffs):
    features = []
    for coeff in coeffs:
        features.append(np.sum(coeff ** 2))  # Energy of the coefficients
        features.append(np.mean(np.abs(coeff)))  # Mean absolute value
        features.append(np.std(coeff))  # Standard deviation
    return features


def calculate_severity(coeffs):
    severity = 0
    for coeff in coeffs:
        energy = np.sum(coeff ** 2)
        severity += energy
    return severity


# Example classifier (replace with real model after training)
classifier = RandomForestClassifier()


def classify_fault(coeffs, classifier):
    features = extract_features_from_dwt(coeffs)
    prediction = classifier.predict([features])
    return prediction


# ------------------------------
# Streamlit App with Advanced Features
# ------------------------------

st.title("Future-Ready Fault Detection System")
st.markdown(
    """
    ## Revolutionizing Fault Detection with IoT and Wavelet Analysis
    This application demonstrates advanced methodologies for fault detection in rotating machinery.
    It leverages **real-time sensor data**, **wavelet transform analysis**, and **interactive visualizations**.
    """
)
st.sidebar.header("Configuration Panel")

# ------------------------------
# Fault Selection and Configurations
# ------------------------------

fault_types = ["Real-Time Sensor Data", "Imbalance", "Misalignment", "Bearing Fault"]
selected_fault = st.sidebar.selectbox("Select Fault Type", fault_types)

wavelet_type = st.sidebar.selectbox("Select Wavelet", ["db4", "haar", "sym5"])
decomposition_level = st.sidebar.slider("Decomposition Level", 1, 6, 4)
sampling_rate = 1600
duration = 1

# Signal Generation Based on Selected Fault
if selected_fault == "Real-Time Sensor Data":
    signal = st.session_state["real_time_signal"]
    t = np.linspace(0, duration, len(signal))  # Ensuring t is always defined
else:
    fault_amp = st.sidebar.slider("Fault Amplitude", 0.1, 1.0, 0.5)
    if selected_fault == "Imbalance":
        t, signal = generate_imbalance_signal(frequency=60, sampling_rate=sampling_rate, duration=duration,
                                              imbalance_amp=fault_amp)
    elif selected_fault == "Misalignment":
        mod_freq = st.sidebar.slider("Modulation Frequency", 1, 10, 5)
        mod_index = st.sidebar.slider("Modulation Index", 0.1, 1.0, 0.5)
        t, signal = generate_misalignment_signal(frequency=60, mod_freq=mod_freq, mod_index=mod_index,
                                                 sampling_rate=sampling_rate, duration=duration)
    elif selected_fault == "Bearing Fault":
        bpfo = st.sidebar.slider("BPFO Frequency", 20, 100, 40)
        t, signal = generate_bearing_fault_signal(bpfo=bpfo, sampling_rate=sampling_rate, duration=duration,
                                                  fault_amp=fault_amp)

# ------------------------------
# Time-Domain Signal Analysis
# ------------------------------

st.subheader("Time-Domain Signal Analysis")
st.markdown(
    """
    **Understanding the Basics:**
    Time-domain analysis helps identify the nature of the fault by studying signal amplitude over time.
    """
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", name="Signal"))
fig.update_layout(title="Time-Domain Signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
st.plotly_chart(fig)

# ------------------------------
# DWT Decomposition Visualization
# ------------------------------

st.subheader("Wavelet Decomposition")
st.markdown(
    """
    **Wavelet Transform:**
    Wavelet analysis is ideal for detecting transient faults and localized disturbances in machinery signals.
    """
)
coeffs = perform_dwt_analysis(signal, wavelet=wavelet_type, level=decomposition_level)
fig = go.Figure()
for i, coeff in enumerate(coeffs):
    fig.add_trace(go.Scatter3d(x=np.arange(len(coeff)), y=np.full(len(coeff), i + 1), z=coeff, mode="lines",
                               name=f"Level {i + 1}"))
fig.update_layout(title="3D Wavelet Coefficients",
                  scene=dict(xaxis_title="Samples", yaxis_title="Level", zaxis_title="Amplitude"))
st.plotly_chart(fig)

# ------------------------------
# Signal Filtering and Frequency Spectrum
# ------------------------------

cutoff_frequency = st.sidebar.slider("Cutoff Frequency", 10, 200, 50)
filtered_signal = butter_filter(signal, cutoff=cutoff_frequency, fs=sampling_rate)

frequencies, psd = welch(filtered_signal, fs=sampling_rate)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=frequencies, y=frequencies, z=psd, mode="markers",
                           marker=dict(size=3, color=psd, colorscale="Viridis", showscale=True)))
fig.update_layout(title="3D Frequency Spectrum",
                  scene=dict(xaxis_title="Frequency (Hz)", yaxis_title="Frequency (Hz)", zaxis_title="Power"))
st.plotly_chart(fig)

# ------------------------------
# Fault Severity and Alerts (Removed display_severity)
# ------------------------------

severity = calculate_severity(coeffs)
if severity > 1000:
    st.error("High Severity Fault!")
elif severity > 500:
    st.warning("Moderate Fault Detected")
