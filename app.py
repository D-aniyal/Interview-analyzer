import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
from faster_whisper import WhisperModel
from openai import OpenAI

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎤 Interview Analyzer")
st.write("Upload an interview audio file to analyze transcript, voice features, and get AI feedback.")

# API Key Input (secure way for user)
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key (optional for AI feedback)", type="password")

# File uploader
audio_file = st.file_uploader("📂 Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save uploaded file
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.read())
    audio_path = "uploaded_audio.wav"

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Waveform visualization
    st.subheader("📊 Waveform")
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # Transcription with Whisper
    st.subheader("📝 Transcription")
    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe(audio_path)
    transcript = " ".join([seg.text for seg in segments])
    st.text_area("Transcript", transcript, height=200)

    # Voice Features
    st.subheader("🎙️ Voice Features")
    duration = librosa.get_duration(y=y, sr=sr)
    words = transcript.split()
    wpm = len(words) / (duration / 60)

    # Pitch analysis
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300)
    avg_pitch = np.nanmean(f0)

    st.write(f"**Duration:** {duration:.2f} sec")
    st.write(f"**Words per minute (WPM):** {wpm:.2f}")
    st.write(f"**Average Pitch (Hz):** {avg_pitch:.2f}")

    # AI Feedback using OpenAI LLM
    llm_analysis = "Not generated (API key not provided)."
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            prompt = f"""
            You are an expert interview coach. Analyze the following transcript and provide:
            1. Overall speaking clarity
            2. Confidence level
            3. Vocabulary richness
            4. Suggestions for improvement

            Transcript:
            {transcript}
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            llm_analysis = response.choices[0].message.content
        except Exception as e:
            llm_analysis = f"Error: {e}"

    st.subheader("🤖 AI Feedback")
    st.write(llm_analysis)

    # Save results
    results = {
        "metadata": {"duration_sec": duration, "words_per_minute": wpm},
        "transcript": transcript,
        "features": {"avg_pitch": float(avg_pitch)},
        "llm_analysis": llm_analysis,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    st.download_button("📥 Download Results JSON", data=json.dumps(results, indent=4), file_name="results.json")
