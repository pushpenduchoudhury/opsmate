import yaml
from pathlib import Path
import config.conf as conf
import streamlit as st
import streamlit_authenticator as stauth


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