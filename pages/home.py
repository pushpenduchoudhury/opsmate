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

    if len(accessible_apps) == 0:
        st.info("You do not have the privilege to view any Apps. Please reach out to the administrator for access...!")
        st.stop()


no_of_apps = len(accessible_apps)
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
            st.image(image = str(Path(conf.ASSETS_DIR, accessible_apps[app_num]["image_icon"])), width = image_width)
            
            # App Title
            st.subheader(accessible_apps[app_num]["name"], divider = "grey", anchor = False)
            
            # App Description
            desc_col = st.columns(1)
            with desc_col[0].container(border = False, height = int(0.18 * tile_height)):
                st.markdown(f'<span style="font-size: 16px; text-align: center;">{accessible_apps[app_num]["description"]}</span>', unsafe_allow_html = True)
            
            # App Launch Button
            if st.button("Launch", key = f"app_{app_num}"):
                st.switch_page(Path(conf.PAGES_DIR, accessible_apps[app_num]["page"]))
                
            app_num += 1