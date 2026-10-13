"""Custom CSS for the navy / white / muted-blue look."""
import streamlit as st

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.mv-title {font-size: 1.9rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0;}
.mv-sub {color: #8FA3C4; margin-top: 0.2rem; margin-bottom: 1.2rem;}
.mv-card {background: #12203A; border: 1px solid #1F3354; border-top: 3px solid #5B8DEF;
          border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;}
.mv-card .label {color: #8FA3C4; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em;}
.mv-card .value {color: #FFFFFF; font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.mv-card .note {color: #8FA3C4; font-size: 0.78rem;}
.mv-demo {background: #C0392B; color: #FFFFFF; font-weight: 700; text-align: center;
          padding: 10px; border-radius: 8px; margin-bottom: 14px; letter-spacing: .04em;}
.mv-step {color: #C9D5EA; padding: 2px 0;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
