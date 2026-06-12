# 👁️ OcularTrack-AI

**A high-performance, real-time bilateral ocular telemetry engine** built on MediaPipe Face Mesh and OpenCV — designed to track iris kinematics, compute angular velocity, and flag anomalous saccadic events with sub-frame latency.

OcularTrack-AI transforms a standard webcam feed into a structured kinematic data stream, suitable for biomedical research prototyping, human-factors monitoring, and industrial vision-safety pipelines.

---

## ✨ Key Features

- **Bilateral Iris Tracking** — Simultaneously tracks both eyes using MediaPipe's 478-point refined face mesh topology (iris landmarks `468` and `473`).
- **Ocular-Centre Coordinate Mapping** — Computes a stable anchor point per eye from medial/lateral canthus landmarks, and maps the iris position into a displacement vector relative to that anchor.
- **Real-Time Angular Velocity Estimation** — Derives signed angular velocity (deg/s) per eye, per frame, using a vectorised 2D cross-product estimator.
- **Anomaly / Saccade Detection** — Configurable velocity threshold flags high-speed ocular events ("ANOMALY DETECTED") with live on-screen alerts.
- **Structured Telemetry Logging** — Streams per-frame metrics (timestamp, target eye, velocity, inter-frame interval, status) to an in-memory log and exports to CSV on shutdown.
- **Live Diagnostic Overlay** — Renders iris markers, anchor markers, velocity HUD, and anomaly banners directly onto the video feed.
- **Fully Configurable via Dataclasses** — Landmark indices, camera parameters, FaceMesh confidence thresholds, and telemetry settings are centralised and type-safe.
- **Structured Logging** — Dual-output (console + rotating file) logging via Python's `logging` module, with graceful camera/resource shutdown handling.

---

## 📐 Mathematical Logic

### 1. Displacement Vector

For each eye, the **ocular centre** (anchor) is computed as the midpoint between the medial and lateral canthus landmarks:

$$
\text{anchor} = \left( \frac{x_{\text{inner}} + x_{\text{outer}}}{2},\ \frac{y_{\text{inner}} + y_{\text{outer}}}{2} \right)
$$

The **displacement vector** $\vec{r}$ from the anchor to the iris centre is:

$$
\vec{r} = (x, y) = (x_{\text{iris}} - x_{\text{anchor}},\ y_{\text{iris}} - y_{\text{anchor}})
$$

### 2. Angular Velocity (Cross-Product Estimator)

Rather than computing `arctan2` every frame, OcularTrack-AI derives angular velocity directly from the 2D cross product between the current displacement vector $\vec{r}$ and the incremental displacement $\Delta \vec{r} = \vec{r}_t - \vec{r}_{t-1}$:

$$
\omega_{\text{rad/s}} = \frac{\vec{r} \times \Delta\vec{r}}{|\vec{r}|^2 \cdot \Delta t} = \frac{x \cdot \Delta y - y \cdot \Delta x}{(x^2 + y^2) \cdot \Delta t}
$$

The result is converted to degrees per second:

$$
\omega_{\text{deg/s}} = \omega_{\text{rad/s}} \times \frac{180}{\pi}
$$

### 3. Numerical Stability Safeguards

* If $|\vec{r}|^2 < r_{sq\_ min}$ (iris sitting on the anchor), $\omega$ is clamped to `0.0` to avoid division-by-zero singularities.
* If $\Delta t \le 0$ (clock anomalies / first frame), $\omega$ is clamped to `0.0`.
* Tracker state (`prev_disp`) is reset whenever face detection is lost, preventing stale deltas from contaminating re-acquisition.

### 4. Anomaly Classification

$$
\text{Status} =
\begin{cases}
\text{ANOMALY DETECTED}, & |\omega_{\text{deg/s}}| > \omega_{\text{threshold}} \\
\text{Normal}, & \text{otherwise}
\end{cases}
$$

---

## 🗂️ Repository Structure

| File | Description |
|---|---|
| `main.py` | Application entry point. Bootstraps structured logging (console + file) and launches the `OcularTelemetryEngine`. |
| `config.py` | Centralised, type-safe configuration via dataclasses: `LandmarkConfig` (immutable MediaPipe landmark indices), `FaceMeshConfig`, `CameraConfig`, `TelemetryConfig`, and the aggregate `EngineConfig`. |
| `engine.py` | Core orchestrator (`OcularTelemetryEngine`). Manages the camera lifecycle, drives MediaPipe FaceMesh inference, dispatches landmark data to trackers, and coordinates rendering and logging. |
| `tracker.py` | `EyeTracker` class — stateful per-eye kinematic engine implementing the cross-product angular velocity formula. Defines the `EyeMetrics` result dataclass and the `build_eye_trackers()` factory. |
| `renderer.py` | `OverlayRenderer` class — isolates all OpenCV drawing logic (iris/anchor markers, velocity HUD, anomaly banners) from the capture loop. |
| `logger.py` | `DataLogger` class — accumulates per-frame telemetry records and exports a structured CSV diagnostic report on shutdown. |
| `requirements.txt` | Pinned Python dependencies for reproducible environments. |
| `ocular_telemetry.log` | *(generated at runtime)* Structured log output from the session. |
| `eye_movement_telemetry_report.csv` | *(generated at runtime)* Exported telemetry diagnostic report. |

---

## 🛠️ Tech Stack

| Component | Purpose |
|---|---|
| **Python 3.10+** | Core language (uses `X \| Y` union type syntax) |
| **MediaPipe** | 478-point refined Face Mesh for iris and canthus landmark detection |
| **OpenCV (`opencv-python`)** | Camera capture, frame pre-processing, and overlay rendering |
| **NumPy** | Vectorised angular velocity computation (dot product, cross product) |
| **pandas** | Structured telemetry accumulation and CSV export |
| **Python `logging`** | Dual-sink (console + file) structured diagnostic logging |

---

## 📷 Hardware Considerations

> **⚠️ Production Note — Camera Selection**
>
> The default configuration targets a standard consumer webcam (`device_index=0`, ~30 FPS), which is suitable for prototyping and demonstration.
>
> For **production-grade deployments** — particularly those targeting **micro-saccade detection**, **clinical-grade ocular kinematics**, or **high-velocity anomaly classification** — a standard RGB webcam introduces motion blur and frame-rate limitations that degrade the accuracy of the $\omega_{\text{deg/s}}$ estimator at high angular velocities.
>
> **Recommendation:** Deploy with a **High-Speed Infrared (IR) global-shutter camera** (≥120 FPS, ideally 200–500 FPS) for:
> - Elimination of motion blur during rapid saccadic movement
> - Consistent illumination independent of ambient lighting conditions
> - Higher temporal resolution for $\Delta t$, improving the precision of the cross-product angular velocity formula
>
> Update `CameraConfig.fps_target`, `frame_width`, and `frame_height` in `config.py` to match your IR sensor's native specifications.

---

## 🚀 Quick Start Guide

### Prerequisites

- Python **3.10** or higher
- A connected camera device (webcam or IR sensor)
- `pip` package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/OcularTrack-AI.git
cd OcularTrack-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Engine

```bash
python main.py
```

### Runtime Controls

| Key | Action |
|---|---|
| `Q` | Terminate the capture loop, export telemetry to CSV, and shut down gracefully |

### Output Artifacts

After a session ends, two files are generated in the working directory:

- `ocular_telemetry.log` — full structured session log
- `eye_movement_telemetry_report.csv` — per-frame telemetry (timestamp, target eye, velocity, interval, status)

---

## 🏭 Industrial Integration Note

OcularTrack-AI's modular architecture — particularly the decoupled `EyeMetrics` data contract produced by `tracker.py` — makes it straightforward to bridge this CV pipeline into **industrial automation and control environments**.

### Integration Pathways

- **Modbus TCP** — The `DataLogger.record()` hot path can be extended with a lightweight Modbus TCP client (e.g. `pymodbus`) to write `omega_deg_s` and `is_anomaly` flags into PLC holding registers in real time, enabling downstream control logic to react to ocular anomaly events.
- **OPC UA** — For SCADA-integrated environments, an OPC UA server layer can expose `EyeMetrics` fields as live tags/nodes, allowing HMI dashboards and historian systems to subscribe to telemetry streams alongside other plant-floor data.
- **Safety / E-Stop Triggers** — The existing `is_anomaly` boolean (derived from `velocity_threshold_deg_s` in `TelemetryConfig`) is a natural trigger point: a sustained anomaly state can be mapped to a discrete output signal driving a **safety relay or E-stop circuit**, supporting operator-attention or fatigue-monitoring safety interlocks.

### Suggested Architecture

```
┌─────────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  OcularTrack-AI      │      │  Protocol Bridge  │      │  PLC / SCADA     │
│  (engine.py loop)    │ ───▶ │  (Modbus / OPC UA)│ ───▶ │  Safety Layer    │
│  → EyeMetrics stream │      │  Register/Tag map │      │  E-Stop / HMI    │
└─────────────────────┘      └──────────────────┘      └─────────────────┘
```

> **Implementation tip:** Hook the protocol bridge into `OcularTelemetryEngine._process_frame()`, immediately after `EyeMetrics` is computed — this keeps the capture loop's latency profile unaffected by network I/O if the bridge writes are non-blocking or queued asynchronously.

---

## 📄 License

Add your preferred license here (e.g. MIT, Apache 2.0).
