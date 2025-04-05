import streamlit as st
import base64

st.title("Slides ")

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

display_pdf("/Users/branndonmarion/Desktop/MIDS/DS 210/w210_snow_intro/slides/slides.pdf")