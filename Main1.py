import numpy as np
import json
from websocket_server import WebsocketServer
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import streamlit as st
import threading

# ------------------------------
# Historical Data Management
# ------------------------------
HISTORICAL_DATA = deque(maxlen=100)  # Store up to 100 data points

# ------------------------------
# Signal Analysis Functions
# ------------------------------
def calculate_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def perform_fft(signal, sampling_rate):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_values = np.abs(np.fft.fft(signal))
    return freqs[:n // 2], fft_values[:n // 2]

def analyze_vibration_data(vibration_data, sampling_rate):
    rms_value = calculate_rms(vibration_data)
    freqs, fft_values = perform_fft(vibration_data, sampling_rate)
    dominant_frequency = freqs[np.argmax(fft_values)]
    return {
        "RMS Value": rms_value,
        "Dominant Frequency": dominant_frequency
    }

# ------------------------------
# Update Historical Data
# ------------------------------
def update_historical_data(analysis_results):
    HISTORICAL_DATA.append({
        "Timestamp": datetime.now(),
        "RMS Value": analysis_results["RMS Value"],
        "Dominant Frequency": analysis_results["Dominant Frequency"]
    })

# ------------------------------
# WebSocket Handler
# ------------------------------
def handle_new_client(client, server):
    print(f"New client connected: {client}")

def handle_client_message(client, server, message):
    """
    Handles incoming messages from WebSocket clients.
    Processes vibration data and sends back analysis results.
    """
    try:
        data = json.loads(message)
        vibration_data = np.array(data["vibration_data"])
        sampling_rate = data["sampling_rate"]
        analysis_results = analyze_vibration_data(vibration_data, sampling_rate)
        update_historical_data(analysis_results)
        server.send_message(client, json.dumps(analysis_results))
    except Exception as e:
        error_message = {"error": str(e)}
        server.send_message(client, json.dumps(error_message))

# ------------------------------
# WebSocket Server Start
# ------------------------------
def start_websocket_server():
    """
    Initializes the WebSocket server to listen for incoming connections.
    """
    server = WebsocketServer(host="0.0.0.0", port=8765)
    server.set_fn_new_client(handle_new_client)
    server.set_fn_message_received(handle_client_message)
    server.run_forever()

# ------------------------------
# Streamlit Application
# ------------------------------
st.title("Induction Motor Vibration Analysis")
st.sidebar.header("Configuration")

if "websocket_thread" not in st.session_state:
    st.session_state.websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    st.session_state.websocket_thread.start()

st.info("Connect to the WebSocket server at ws://<your-server-ip>:8765 to send vibration data for analysis.")

if HISTORICAL_DATA:
    st.subheader("Historical Trends")
    timestamps = [entry["Timestamp"] for entry in HISTORICAL_DATA]
    rms_values = [entry["RMS Value"] for entry in HISTORICAL_DATA]
    dominant_frequencies = [entry["Dominant Frequency"] for entry in HISTORICAL_DATA]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(timestamps, rms_values, marker="o", label="RMS Value")
    ax[0].set_title("Historical RMS Trend")
    ax[0].set_ylabel("RMS Value")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(timestamps, dominant_frequencies, marker="o", label="Dominant Frequency (Hz)", color="orange")
    ax[1].set_title("Historical Dominant Frequency Trend")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].grid()
    ax[1].legend()

    st.pyplot(fig)
