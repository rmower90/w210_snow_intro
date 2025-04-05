import streamlit as st
import os
import base64
from PIL import Image
from pdf2image import convert_from_path

# Write title 
st.title("Modeling/Training Validation")

# Function to display a local PDF file
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# Path to images folder
image_folder = "/Users/branndonmarion/Desktop/MIDS/DS 210/w210_snow_intro/images"

# Dropdown for selecting elevation bin
category_options = ['Elevation 1', 'Elevation 2', 'Elevation 3', 'Elevation 4', 'Elevation 5', 'Elevation 6', 'Elevation 7']
selected_category = st.selectbox('Select Category', category_options)

# Extract the bin number from selection
bin_number = category_options.index(selected_category) + 1
pdf_filename = f"bin{bin_number}.pdf"
pdf_path = os.path.join(image_folder, pdf_filename)

# Display selected PDF
display_pdf(pdf_path)

# Load your images
img1 = Image.open(pdf_path)  # Main image (left)
img2 = Image.open(pdf_path)  # Top right image
img3 = Image.open(pdf_path)  # Bottom right image

# Layout with columns
col1, col2 = st.columns([2, 1])  # Wider left column

# Left column: one big image
with col1:
    st.image(img1, caption="Main Image", use_column_width=True)

# Right column: two stacked images
with col2:
    st.image(img2, caption="Top Image", use_column_width=True)
    st.image(img3, caption="Bottom Image", use_column_width=True)
