import streamlit as st
import numpy as np
import pywt
from scipy.signal import welch
import plotly.graph_objects as go
import websocket
import threading
import json
import time

# ------------------------------
# 1. Batch Initialization
# ------------------------------

# Initialize session states
if "data_buffer" not in st.session_state:
    st.session_state["data_buffer"] = []
if "last_update_time" not in st.session_state:
    st.session_state["last_update_time"] = time.time()
if "ws_status" not in st.session_state:
    st.session_state["ws_status"] = "Disconnected"
if "stop_thread" not in st.session_state:
    st.session_state["stop_thread"] = False

# ------------------------------
# 2. WebSocket Integration
# ------------------------------

def process_batch(data_buffer):
    """Process a batch of data for visualization."""
    x_vals = [data["x"] for data in data_buffer]
    y_vals = [data["y"] for data in data_buffer]
    z_vals = [data["z"] for data in data_buffer]
    return np.array(x_vals), np.array(y_vals), np.array(z_vals)

def on_message(ws, message):
    """Handle incoming WebSocket messages."""
    try:
        data = json.loads(message)
        st.session_state["data_buffer"].append(data)

        # Process and visualize if buffer is large enough
        if len(st.session_state["data_buffer"]) >= 160:  # Process every 0.1 seconds
            batch = st.session_state["data_buffer"]
            x, y, z = process_batch(batch)
            visualize_data(x, y, z)
            st.session_state["data_buffer"] = []  # Clear buffer
    except json.JSONDecodeError:
        st.warning("Received malformed data from WebSocket.")

def on_error(ws, error):
    """Handle WebSocket errors."""
    st.error(f"WebSocket error: {error}")
    st.session_state["ws_status"] = "Error"

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket closure."""
    st.warning("WebSocket closed.")
    st.session_state["ws_status"] = "Disconnected"

def websocket_thread():
    """WebSocket thread to handle real-time data."""
    st.session_state["ws_status"] = "Connecting"
    try:
        ws = websocket.WebSocketApp(
            "ws://192.168.137.114/ws",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        st.session_state["ws_status"] = "Connected"
        ws.run_forever()
    except Exception as e:
        st.error(f"WebSocket connection failed: {e}")
        st.session_state["ws_status"] = "Disconnected"

# Start WebSocket thread
if not st.session_state.get("thread_started", False):
    thread = threading.Thread(target=websocket_thread, daemon=True)
    thread.start()
    st.session_state["thread_started"] = True

# ------------------------------
# 3. Visualization Functions
# ------------------------------

def visualize_data(x, y, z):
    """Update charts with new batch of data."""
    if time.time() - st.session_state["last_update_time"] > 0.1:
        st.session_state["last_update_time"] = time.time()

        # Plot time-domain signals
        st.subheader("Time-Domain Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=x, mode="lines", name="X-Axis"))
        fig.add_trace(go.Scatter(y=y, mode="lines", name="Y-Axis"))
        fig.add_trace(go.Scatter(y=z, mode="lines", name="Z-Axis"))
        fig.update_layout(title="Real-Time Data", xaxis_title="Samples", yaxis_title="Amplitude")
        st.plotly_chart(fig)

        # Frequency spectrum using Welch's method
        st.subheader("Frequency Spectrum")
        freq_x, psd_x = welch(x, fs=1600)
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=freq_x, y=psd_x, mode="lines", name="X-Axis"))
        fig_freq.update_layout(title="Frequency Spectrum (X-Axis)", xaxis_title="Frequency (Hz)", yaxis_title="Power")
        st.plotly_chart(fig_freq)

# ------------------------------
# 4. Streamlit Interface
# ------------------------------

st.title("Optimized Real-Time Vibration Monitoring")
st.sidebar.header("Configuration Options")
st.sidebar.info("Ensure the Wi-Fi module is connected and streaming data.")
st.sidebar.write(f"WebSocket Status: {st.session_state['ws_status']}")
