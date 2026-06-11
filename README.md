# OcularTrack-AI 👁️🚀

A high-performance, real-time ocular telemetry and anomaly detection engine powered by Computer Vision. This system builds a localized coordinate system for tracking pupil dynamics, measuring angular velocity, and capturing micro-saccadic anomalies via mathematical vector analysis.

---

## 🌟 Key Features

* **Localized Coordinate Mapping:** Anchors a local $(0,0)$ reference system based on eye corner landmarks to cancel out global head motion artifacts.
* **Angular Velocity Tracking:** Continuously monitors the rotational velocity of the pupil ($\omega$ in $\text{deg/s}$) across sequential frames using vector cross-product differentiation.
* **Real-Time Anomaly Detection:** Instantly captures rapid eye movements (Saccades/Spikes) exceeding a threshold of **500°/s**, triggering immediate visual alerts and database logging.
* **Automated Telemetry Logging:** Generates comprehensive, production-ready `.csv` reports and structured `.log` files documenting precise metrics for every frame.

---

## 📊 Technical Architecture & Logic

The engine operates on a strict **Kinematics & Spatial Geometry Pipeline**:

```text
[Camera Frame Input] 
         │
         ▼
[Isolate Eye Regions & Establish Corner Reference Point (0,0)]
         │
         ▼
[Track Pupil Center Vector (x, y)]
         │
         ▼
[Calculate Rotational Velocity (ω) relative to Frame Time (dt)]
         │
         ▼
 ┌───────┴───────┐
 │ ω > 500°/s ?  │
 └───────┬───────┘
         ├───► YES ──► Trigger [ANOMALY DETECTED] Alert & Log CSV/Log
         └───► NO  ──► Status: [Normal] Continuous Telemetry Stream
