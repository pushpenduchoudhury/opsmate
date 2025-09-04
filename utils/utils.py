import os
import yaml
import pyttsx3
import tempfile
from groq import Groq
import streamlit as st
from pathlib import Path
import config.conf as conf
import speech_recognition as sr
import streamlit_authenticator as stauth
from dotenv import load_dotenv
load_dotenv()

client = Groq()

def get_config():
    with open(Path(conf.CONFIG_DIR, "creds.yaml")) as file:
        config = yaml.safe_load(file)
    return config

def hash_password(password) -> str:
    hashed_password = stauth.Hasher.hash(password = password)
    return hashed_password

def check_password(email_id, str_password) -> bool:
    config = get_config()
    hashed_password = config["credentials"]["usernames"][email_id]['password']
    is_valid = stauth.Hasher.check_pw(password = str_password, hashed_password = hashed_password)
    return is_valid

def apply_css(file):
    CSS_FILE = Path(conf.CSS_DIR, file)
    with open(CSS_FILE) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html = True)

def get_temp_file(suffix, data = None):
    with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as tmp:
        if data is not None:
            tmp.write(data)
        tmp_path = tmp.name
    return tmp_path

def audio_input(adjust_for_ambient_noise:bool = False):
    with sr.Microphone() as source:
        recognizer = sr.Recognizer()
        if adjust_for_ambient_noise:
            with st.spinner(":grey[Adjusting for ambient noise...]"):
                recognizer.adjust_for_ambient_noise(source)
        with st.spinner(":grey[Listening...]"):
            audio = recognizer.listen(source)
    return audio.get_wav_data()

def speech_to_text(audio_data):
    tmp_path = get_temp_file(suffix = ".wav", data = audio_data)
    with open(tmp_path, "rb") as file:
        transcript = client.audio.transcriptions.create(model = conf.SPEECH_TO_TEXT_MODEL, file = file, language = "en")
    os.remove(tmp_path)
    return transcript.text

def text_to_speech_groq(text, voice):
    response = client.audio.speech.create(input = text, model = conf.TEXT_TO_SPEECH_MODEL, voice = voice, response_format = "wav")
    audio = response.read()
    return audio

def text_to_speech_native(text, voice = None):
    engine1 = pyttsx3.init()
    tmp_path = get_temp_file(suffix = ".wav")
    rate = engine1.getProperty('rate')
    engine1.setProperty('rate', rate - 50)
    engine1.save_to_file(text = text, filename = tmp_path)
    engine1.runAndWait()
    audio_bytes = None
    with open(tmp_path, 'rb') as f:
        audio_bytes = f.read()
    os.remove(tmp_path)
    return audio_bytes