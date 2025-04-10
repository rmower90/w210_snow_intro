import streamlit as st
import streamlit.components.v1 as components
import base64

st.set_page_config(page_title="Background Presentation", layout="wide")
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

display_pdf("slides/slides.pdf")


embed_url = "https://docs.google.com/presentation/d/1cFgZvaY9bqWV2Kb8Bwpyv8krJZoJ6usgwN9UAGztpHM/edit#slide=id.p"
components.iframe(embed_url, width=2200, height=1200)

from PIL import Image
import os

# --- SETUP: Load images ---
image_folder = "slides/indiv"
images = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# --- STATE: Track slide index ---
if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0

# --- LAYOUT: Prev / Next buttons ---
col1, col2, col3 = st.columns([1, 6, 1])

# --- DISPLAY: Current slide ---
current_image = Image.open(images[st.session_state.slide_idx])
st.image(current_image, caption=f"Slide {st.session_state.slide_idx + 1}/{len(images)}", use_container_width=True)

with col1:
    if st.button("⬅️ Prev"):
        st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(images)

with col3:
    if st.button("Next ➡️"):
        st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(images)
