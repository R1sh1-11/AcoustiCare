# AcoustiCare - OR Soundscape Risk Detector

## 🏥 Overview

Operating rooms are acoustically complex environments where excessive alarms, overlapping speech, and high noise levels can contribute to cognitive overload and potential safety risks.

**AcoustiCare** is a real-time audio analysis tool that quantifies environmental stress in the operating room by analyzing sound patterns and producing an explainable risk score.

The system identifies key contributors such as:

* Speech density
* Alarm/equipment activity
* Overall loudness
* Acoustic burstiness (interruptions)

Our goal is to provide surgical teams with situational awareness about when the environment may be drifting into a high-risk state.

---

## ✨ Features

* 🎧 Audio upload and analysis
* 📊 Real-time cognitive load / risk scoring
* 🔍 Explainable “top contributors” output
* 📈 Timeline visualization of sound activity
* ⚙️ Adjustable risk weighting

---

## 🧠 How It Works

1. Audio is split into short time windows.
2. A pretrained audio model (YAMNet) classifies sound events.
3. The system estimates:

   * speech presence
   * alarm/equipment presence
   * loudness
   * burst frequency
4. These signals are combined into an interpretable risk score.

> ⚠️ This is a research prototype and not a medical device.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/or-soundscape-risk.git
cd or-soundscape-risk
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open: http://localhost:8501

---

## 🧪 Demo Tips

For best results, test with audio that includes:

* overlapping speech
* alarm or beep sounds
* varying noise levels

---

## 👥 Team

* **Jennet Ylyasova** — Medical Lead
* **Rishi Nalam** — Engineer
* **Emiliano Chahin** — Assistant Engineer

*(Davidson Hackathon 2026)*

---

## ⚠️ Limitations

* Uses general audio classification (not OR-specific training)
* Not validated for clinical decision-making
* Intended for demonstration and research purposes only

---

## 🔮 Future Work

* OR-specific model fine-tuning
* Integration with OR workflow phases
* Multimodal inputs (vitals, HRV, etc.)
* Clinical validation studies

---

## 📜 License

MIT License
