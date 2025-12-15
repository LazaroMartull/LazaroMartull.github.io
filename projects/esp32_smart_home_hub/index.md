# ESP32 Smart Home Hub (AWS IoT + Streamlit)

## Overview
This project implements an end-to-end IoT smart home automation system using an ESP32 microcontroller, secure cloud communication, and a real-time web dashboard.

The system collects live sensor data, applies automation logic at the edge and in the cloud, and allows manual control through an interactive interface.

## Problem & Goal
The goal was to design a production-style IoT system that:

- integrates multiple sensors and actuators  
- communicates securely with the cloud  
- supports real-time monitoring and control  
- follows established IoT architecture principles  

## What I Built
- ESP32 firmware integrating temperature, humidity, motion, touch, and light sensors  
- Secure MQTT communication with AWS IoT Core using TLS and certificate-based authentication  
- Automation logic for climate control and lighting with manual override  
- A Streamlit dashboard for real-time data visualization and device control  
- A modular architecture aligned with the IoT World Forum 7-layer model  

## Tools & Technologies
- **Embedded:** ESP32 (C/C++)  
- **Cloud:** AWS IoT Core (MQTT over TLS)  
- **Dashboard:** Python, Streamlit  
- **Visualization:** Plotly  
- **Protocols:** MQTT, HTTPS  

## Key Takeaways
- Designed a secure cloud-to-device IoT pipeline  
- Balanced local edge logic with cloud-based control  
- Worked with real-time data streams and system monitoring  
- Gained hands-on experience with enterprise IoT security concepts  

## Files
- 📄 **Final Report:** [report.pdf](./report.pdf)  
- 🖥️ **Dashboard Application:** [app.py](./app.py)  
- ☁️ **MQTT / Cloud Client:** [mqtt_client.py](./mqtt_client.py)  
- 📦 **Dependencies:** [requirements.txt](./requirements.txt)
