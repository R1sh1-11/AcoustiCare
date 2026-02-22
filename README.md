# 🎙️ AcoustiCare: OR Soundscape Risk Sentinel

## 🏥 Overview
Operating rooms are acoustically complex environments. Between bone saws, overlapping conversations, and constant monitor beeps, surgeons suffer from severe **cognitive overload** and **alarm fatigue**. 

**AcoustiCare** is an Edge AI-powered acoustic sentinel that acts as a "Check Engine Light" for the surgical team. It analyzes sound patterns in real-time without sending sensitive patient audio to the cloud, producing an explainable **Surgical Risk Index (SRI)** to prevent communication breakdowns before they cause surgical errors.

## ✨ Key Features
* **🔴 Live OR Monitor (`live_dashboard.py`):** Real-time acoustic analysis tracking live volume, speech density, and alarm probabilities using a browser-based dashboard.
* **📂 Post-Op Audit Dashboard (`dashboard.py`):** A secure file-uploader for hospital administrators to review post-operative acoustic logs and identify environmental stressors.
* **🧠 Advanced Spike Detection:** Uses `scipy.signal` to track actual *Alarm Spikes per Minute*, moving beyond simple probability to clinical-grade event tracking.
* **📥 Immutable Audit Logs:** Automatically logs "Critical Risk Events" (SRI > 60) with timestamp and primary stressor, exportable to CSV for post-op review.

## 🛠️ The Tech Stack
* **Frontend:** Streamlit
* **Edge AI:** TensorFlow Hub (YAMNet) for localized, HIPAA-compliant audio classification
* **Audio Processing:** PyAudio, Librosa
* **Math & Data:** NumPy, Pandas, SciPy

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone [https://github.com/R1sh1-11/AcoustiCare.git](https://github.com/R1sh1-11/AcoustiCare.git)
cd AcoustiCare
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Applications**
Because we prioritize stability and security, AcoustiCare is split into two independent modules:

*To run the Post-Op Audit Dashboard (Static Files):*
```bash
streamlit run dashboard.py
```

*To run the Live OR Monitor (Real-time Mic):*
```bash
streamlit run live_dashboard.py
```

## 🧪 Demo Tips
For the best evaluation, test the app with audio that includes:
* Overlapping speech
* Sudden, loud equipment alarms
* Periods of chaotic, volatile noise (rustling, dropping items)

## 👥 The Team
* **Jennet Ylyasova** — Medical Lead
* **Rishi Nalam** — Engineer
* **Emiliano Chahin** — Engineer
*(Built for Hack @ Davidson 2026)*

## 🔮 Future Work
* **Federated Learning:** Train the YAMNet model on specific medical sounds across multiple institutions without ever sharing raw audio data.
* **Smart OR Integration:** Interface with IoT devices to automatically mitigate noise (e.g., automatically dimming non-essential lights) when SRI thresholds are exceeded.
* **Visual Stress Analysis:** Incorporate computer vision to correlate acoustic stress with surgeon micro-expressions and ergonomics.
