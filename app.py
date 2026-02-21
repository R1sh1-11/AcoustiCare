import streamlit as st
from live_dashboard import render_live
from dashboard import render_static

st.set_page_config(page_title="AcoustiCare", layout="wide")

tab1, tab2 = st.tabs(["🎙️ Live OR Sentinel", "📂 Post-Op Audit"])

with tab1:
    render_live()

with tab2:
    render_static()