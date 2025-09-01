import math
import streamlit as st
from utils import utils
from pathlib import Path
import config.conf as conf
from dotenv import load_dotenv
load_dotenv()

utils.apply_css("home.css")

st.set_page_config(
        page_title = "IT Support Desk",
        page_icon = "🕵🏻",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

st.header("IT Support Desk", anchor = False, divider = "red")

apps = [
    {"name": "NextGen OpsMate",
     "description": "The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.",
     "page": "opsmate.py",
     "image_icon": "opsmate.gif",
    },
    {"name": "Incident Analytics",
     "description": "Incident ticket analytics provide insights into the volume, types, resolution times, and trends of IT support issues. This data helps identify bottlenecks, improve processes, and optimize resource allocation for faster and more efficient problem resolution.",
     "page": "analytics.py",
     "image_icon": "analytics.png",
    },
]

no_of_apps = len(apps)
app_grid_cols = 4
app_grid_rows = math.ceil(no_of_apps/app_grid_cols)
tile_height = 285
image_width = 65


app_num = 0
for row in range(app_grid_rows):
    st_cols = st.columns(app_grid_cols)
    for col in range(app_grid_cols):
        if app_num > no_of_apps - 1:
            break
        with st_cols[col].container(border = True, height = tile_height):
            
            # Image
            st.image(image = str(Path(conf.ASSETS_DIR, apps[app_num]["image_icon"])), width = image_width)
            
            # App Title
            st.subheader(apps[app_num]["name"], divider = "grey", anchor = False)
            
            # App Description
            desc_col = st.columns(1)
            with desc_col[0].container(border = False, height = int(0.18 * tile_height)):
                st.markdown(f'<span style="font-size: 16px; text-align: center;">{apps[app_num]["description"]}</span>', unsafe_allow_html = True)
            
            # App Launch Button
            if st.button("Launch", key = f"app_{app_num}"):
                st.switch_page(Path(conf.PAGES_DIR, apps[app_num]["page"]))
                
            app_num += 1