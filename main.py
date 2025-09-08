import uuid
import streamlit as st
from utils import utils
from pathlib import Path
import streamlit_authenticator as stauth
from config import conf 

config = utils.get_config()

authenticator = stauth.Authenticate(
    credentials = config['credentials'],
    cookie_name = config['cookie']['name'],
    cookie_key = config['cookie']['key'],
    cookie_expiry_days = config['cookie']['expiry_days']
)

st.markdown(
    f"""
<style>
    .st-emotion-cache-10p9htt:before {{
        content: "𖡎 IT Support Desk";
        font-weight: bold;
        font-size: xx-large;
    }}
</style>""",
        unsafe_allow_html = True,
    )

# Set Page Config
st.set_page_config(
    page_title = "IT Support Desk",
    page_icon = "🕵🏻",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Initialize chat session in streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def clear_session_state_on_logout():
    """Clears relevant items from st.session_state upon logout."""
    keys_to_clear = ["authentication_status", "chat_messages"] 
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

authenticator.login(location = 'main')

if st.session_state.get('authentication_status'):
    
    with st.sidebar:
        col1, col2 = st.columns([0.8, 0.2])
        col1.markdown(f""":grey[*Welcome,*] <br> **{st.session_state.get('name')} :grey[({','.join(st.session_state.get('roles'))})]**""", unsafe_allow_html = True)
        
        with col2:
            authenticator.logout(button_name = ":material/logout:")
    
    apps = conf.APPS
    user_role = st.session_state.get("roles")
    accessible_apps = []
    
    if user_role is not None:
        if "admin" in user_role:
            accessible_apps = apps
        else:
            for app in apps:
                if len(list(set(user_role).intersection(set(app["access_privilege_role"])))) > 0:
                    accessible_apps.append(app)
    
    def get_streamlit_pages():
        pages = []
        for app in accessible_apps:
            page = st.Page(Path(conf.PAGES_DIR, app["page"]), title = app["name"], icon = app["page_icon"])
            pages.append(page)
        return pages
    
    pages = {
        "🏠︎ Home": [
            st.Page(Path(conf.PAGES_DIR, "home.py"), title = "Apps", icon = ":material/home:", default = True),
        ],
        "♕ Apps": get_streamlit_pages(),
        "❔ Help": [
            st.Page(Path(conf.PAGES_DIR, "about.py"), title = "About", icon = ":material/info:"),
        ]
    }
    
    page = st.navigation(pages, position = "top", expanded = True)
    page.run()
    
    
elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
    
elif st.session_state.get('authentication_status') is None:
    st.info('ⓘ Please enter your username and password')
    clear_session_state_on_logout()

