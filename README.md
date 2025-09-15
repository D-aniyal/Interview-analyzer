# 🎤 Interview Analyzer

An AI-powered mini project to analyze interview audio recordings.  
It can transcribe speech, extract voice features, and optionally provide **AI feedback** using OpenAI GPT models.

---

## 🚀 Features
- 📂 Upload audio file (`.wav`, `.mp3`, `.m4a`)
- 📝 Transcribe speech using Whisper (faster-whisper)
- 🎙️ Extract voice features:
  - Duration
  - Words per minute (WPM)
  - Average pitch
- 🤖 AI Feedback (clarity, confidence, vocabulary) with OpenAI API
- 📥 Export results as JSON

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) - UI
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Speech-to-text
- [Librosa](https://librosa.org/) - Audio processing
- [OpenAI](https://platform.openai.com/) - LLM feedback
- Python, NumPy, Matplotlib

---

## 📦 Installation (Run Locally)
```bash
# Clone repo
git clone https://github.com/your-username/interview-analyzer.git
cd interview-analyzer

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
