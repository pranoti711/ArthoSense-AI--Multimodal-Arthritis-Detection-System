"""
Voice Utilities for Arthritis GUI
Live microphone recording + STT + voice analysis integration.
"""

import os
import tempfile
import io
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import speech_recognition as sr
import pyaudio
from pydub import AudioSegment
import numpy as np
from modules.voice_model.predict import predict as predict_voice

# RTC config
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VoiceRecorder:
    """WebRTC audio recorder."""
    def __init__(self):
        self.audio_bytes = None
    
    def receive_audio(self, frame):
        self.audio_bytes = frame.to_ndarray().tobytes()
        return frame
    
    def get_audio_file(self):
        if self.audio_bytes is None:
            return None
        audio = AudioSegment(
            self.audio_bytes,
            frame_rate=16000,
            sample_width=2,
            channels=1
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio.export(tmp.name, format="wav")
            return tmp.name

def audio_to_mp3(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    mp3_path = audio_path.replace('.wav', '.mp3')
    audio.export(mp3_path, format="mp3")
    return mp3_path

def stt_transcribe(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text.strip()
    except:
        return None

def record_and_analyze():
    st.subheader("🎤 **Live Voice Analysis** (Click to start)")
    
    recorder = VoiceRecorder()
    
    ctx = webrtc_streamer(
        key="arthritis-voice-mic",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=recorder.receive_audio,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": False,
            "audio": {
                "sampleRate": 16000,
                "channelCount": 1,
                "echoCancellation": True,
                "noiseSuppression": True
            }
        }
    )
    
    if ctx.state.playing:
        st.info("🎙️ **Recording...** Speak normally for 10-30 seconds")
        st.balloons()
    
    # Analyze button
    if st.button("🔬 **Analyze Voice**", type="primary", disabled=recorder.audio_bytes is None):
        audio_path = recorder.get_audio_file()
        if audio_path:
            st.audio(audio_path)
            
            # Voice model prediction
            label, confidence = predict_voice(audio_path)
            os.unlink(audio_path)
            
            col1, col2 = st.columns([1,1])
            col1.metric("🎯 Prediction", label.title())
            col2.metric("📊 Confidence", f"{confidence:.1%}")
            
            # Risk feedback
            risk_color = "green" if label == "normal" else "orange" if label == "fatigue" else "red"
            risk_msg = f"**{label.title()} patterns detected**"
            st.markdown(f'<div style="background-color: {"#d4edda" if risk_color=="green" else "#fff3cd" if risk_color=="orange" else "#f8d7da"}; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {"#28a745" if risk_color=="green" else "#ffc107" if risk_color=="orange" else "#dc3545"};">{risk_msg}</div>', unsafe_allow_html=True)
            
            st.session_state.voice_result = {"label": label, "confidence": confidence}
            st.rerun()
    
    if st.session_state.get("voice_result"):
        st.success("✅ Analysis saved to session")
    
    return st.session_state.get("voice_result", {"label": None, "confidence": 0})

# Test page
if __name__ == "__main__":
    st.header("Voice Test Page")
    record_and_analyze()
