# app.py
import streamlit as st
import time
from datetime import datetime

# Import from your modules
from utils import (
    initialize_session_state, load_css, load_aws_config,
    create_gauge, create_timeseries_chart
)
from mqtt_client import SmartHomeMQTTClient

# --- Page Configuration ---
st.set_page_config(page_title="Smart Home Hub", page_icon="🏠", layout="wide")
load_css()
initialize_session_state()

# --- Queue Processing ---
def process_mqtt_updates():
    """Reads the queue and updates session_state in the main thread."""
    if 'mqtt_client' in st.session_state and st.session_state.mqtt_client:
        updates = st.session_state.mqtt_client.get_updates()
        for update_type, payload in updates:
            if update_type == 'log':
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                st.session_state.debug_log.append(f"[{timestamp}] {payload}")
            elif update_type == 'raw_payload':
                st.session_state.raw_payloads.append(payload)
            elif update_type == 'data':
                st.session_state.device_data.update(payload)
                st.session_state.last_update_time = datetime.now()
                st.session_state.data_history.append({'timestamp': datetime.now(), **payload})
            elif update_type == 'status':
                st.session_state.mqtt_connected = payload

# Initial processing
process_mqtt_updates()
data = st.session_state.device_data

# --- Sidebar ---
with st.sidebar:
    st.header("🏠 Smart Home Hub")

    if not st.session_state.mqtt_connected:
        if st.button("🔌 Connect to AWS IoT", type="primary"):
            config = load_aws_config()
            if config:
                st.session_state.mqtt_client = SmartHomeMQTTClient(config)
                st.session_state.mqtt_client.connect()
    else:
        if st.button("🔌 Disconnect from AWS IoT"):
            if st.session_state.mqtt_client:
                st.session_state.mqtt_client.disconnect()

    status = "🟢 Connected" if st.session_state.mqtt_connected else "🔴 Disconnected"
    st.metric("Connection Status", status)
    if st.session_state.last_update_time:
        secs_ago = (datetime.now() - st.session_state.last_update_time).total_seconds()
        st.write(f"Last update: {int(secs_ago)}s ago")
    else:
        st.write("No data received yet.")

    st.divider()
    st.header("⚙️ System Reset")
    if st.button("🔄 Reset All to Auto Mode"):
        if st.session_state.mqtt_client:
            st.session_state.mqtt_client.publish({"command": "system_reset"})
            st.toast("✅ Reset command sent!")

# --- Main Page Layout ---
st.title("Live Smart Home Dashboard")
st.markdown("Real-time sensor data and controls for your ESP32 hub.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎛️ Controls", "📈 System Health", "🔍 Debug"])

# (The content of the tabs remains the same as the previous version)
with tab1:
    st.header("Current Conditions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{data['temp']:.1f}°F")
    col2.metric("💧 Humidity", f"{data['humid']:.1f}%")
    col3.metric("⚡ Power Usage", f"{data['power_watts']:.1f}W")
    uptime_hours = data['uptime'] / 3600
    col4.metric("⏱️ Uptime", f"{uptime_hours:.2f} hrs")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_gauge(data['temp'], "Temperature", "°F", [50, 100], ['#333', '#FF6347']), use_container_width=True)
    with col2:
        st.plotly_chart(create_gauge(data['humid'], "Humidity", "%", [0, 100], ['#333', '#1E90FF']), use_container_width=True)

    st.divider()
    st.header("Data Trends")
    st.plotly_chart(create_timeseries_chart(), use_container_width=True)

with tab2:
    st.header("Device Controls")
    client = st.session_state.get('mqtt_client')

    with st.container(border=True):
        st.subheader("🌡️ Air Conditioner")
        # ... (rest of the tab content is identical)
        if client:
            is_manual = data['manual_ac']
            mode = "Manual" if is_manual else "Auto"
            st.info(f"Current Mode: **{mode}** | Status: **{data['ac']}**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Switch to Auto Mode", disabled=not is_manual):
                    client.publish({"command": "ac_control", "mode": "auto"})
                if st.button("Switch to Manual Mode", disabled=is_manual):
                    client.publish({"command": "ac_control", "mode": "manual", "state": data['ac']})
            with col2:
                new_setpoint = st.slider("Temp Setpoint (°F)", 60, 85, int(data['ac_setpoint']), key="ac_setpoint")
                if st.button("Update Setpoint"):
                    client.publish({"command": "ac_setpoint", "temperature": new_setpoint})
            if is_manual:
                state = st.selectbox("Set AC State", ["OFF", "COOL", "HEAT"], index=["OFF", "COOL", "HEAT"].index(data['ac']))
                if st.button("Set AC State"):
                    client.publish({"command": "ac_control", "mode": "manual", "state": state})

    with st.container(border=True):
        st.subheader("💡 Lights")
        # ... (rest of the tab content is identical)
        if client:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Indoor Light (Motion)**")
                is_manual_indoor = data['manual_indoor']
                st.info(f"Mode: {'Manual' if is_manual_indoor else 'Auto'} | Status: **{data['indoor']}**")
                if st.button("Toggle Indoor Manual/Auto"):
                    new_mode = "manual" if not is_manual_indoor else "auto"
                    client.publish({"command": "indoor_light", "mode": new_mode, "state": data['indoor'] == 'ON'})
                if is_manual_indoor:
                    if st.button("Turn Indoor Light ON" if data['indoor'] == 'OFF' else "Turn Indoor Light OFF"):
                        client.publish({"command": "indoor_light", "mode": "manual", "state": not (data['indoor'] == 'ON')})
            with col2:
                st.markdown("**Outdoor Light (Touch)**")
                is_manual_outdoor = data['manual_outdoor']
                st.info(f"Mode: {'Manual' if is_manual_outdoor else 'Auto'} | Status: **{data['outdoor']}**")
                if st.button("Toggle Outdoor Manual/Auto"):
                    new_mode = "manual" if not is_manual_outdoor else "auto"
                    client.publish({"command": "outdoor_light", "mode": new_mode, "state": data['outdoor'] == 'ON'})
                if is_manual_outdoor:
                    if st.button("Turn Outdoor Light ON" if data['outdoor'] == 'OFF' else "Turn Outdoor Light OFF"):
                        client.publish({"command": "outdoor_light", "mode": "manual", "state": not (data['outdoor'] == 'ON')})

with tab3:
    st.header("System Health & Statistics")
    # ... (rest of the tab content is identical)
    col1, col2, col3 = st.columns(3)
    col1.metric("Wi-Fi Signal (RSSI)", f"{data['rssi']} dBm")
    col2.metric("Free Memory", f"{data['mem'] / 1024:.1f} KB")
    col3.metric("Secure Tunnel", "🟢 Active" if data['tunnel_active'] else "🔴 Inactive")
    st.divider()
    st.subheader("Operation Counts")
    col1, col2, col3 = st.columns(3)
    col1.metric("AC Operations", data['ac_ops'])
    col2.metric("Motion Triggers", data['motion_count'])
    col3.metric("Touch Triggers", data['touch_count'])
    st.divider()
    st.subheader("Reliability")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sensor Read Errors", data['sensor_errors'])
    col2.metric("Wi-Fi Reconnects", data['wifi_reconnects'])
    col3.metric("AWS Reconnects", data['aws_reconnects'])


with tab4:
    st.header("Debug Information")
    # ... (rest of the tab content is identical)
    st.subheader("Raw MQTT Payloads")
    st.code('\n'.join(st.session_state.raw_payloads), language='json')
    st.subheader("Application Log")
    st.code('\n'.join(st.session_state.debug_log))
    st.subheader("Current Device State")
    st.json(data)


# --- Auto-refresh loop ---
# Only rerun if we have a client instance, connected or not
if st.session_state.get('mqtt_client'):
    time.sleep(1) # Refresh every 1 second
    st.rerun()
