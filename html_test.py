import streamlit as st

st.set_page_config(
    page_title="HTML Test",
)

st.html(
    """
    <div style="
        padding: 24px;
        border-radius: 16px;
        background: #111827;
        color: white;
        font-size: 28px;
        font-weight: 700;
    ">
        AI Business Analytics
    </div>
    """
)