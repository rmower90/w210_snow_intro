import streamlit as st
from PIL import Image

# Load images (update the file paths accordingly)
header_image = Image.open("../w210_snow_intro/images/sierra-nevada.webp")
swe_diagram = Image.open("../w210_snow_intro/images/Overview/less_is_more.png")

# Configure the page: wide layout and an expanded sidebar by default
st.set_page_config(page_title="SWE Prediction Project", layout="wide", initial_sidebar_state='expanded')

# --- Custom CSS for a professional scientific template ---
st.markdown("""
    <style>
        /* Base container styling for a clean look */
        .block-container {
            padding: 2rem 5rem;
            max-width: 1200px;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        /* Header styling */
        .header {
            text-align: center;
            border-bottom: 1px solid #e1e1e1;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
        }
        .header h1 {
            font-size: 3rem;
            color: #2c3e50;
            margin: 0;
        }
        .header p {
            font-size: 1.3rem;
            color: #34495e;
            margin-top: 0.5rem;
        }
        /* Sidebar styling: position sidebar on the right */
        [data-testid="stSidebar"] {
            right: 0;
            left: auto;
            padding: 1rem;
        }
        /* Section and card styling */
        .section {
            margin-bottom: 2rem;
            line-height: 1.6;
            font-size: 1.1rem;
            color: #555;
        }
        .card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        /* List style adjustments */
        ul {
            padding-left: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section with Image ---
st.markdown("""
<div class="header">
    <h1>SWE Prediction Project</h1>
    <p>Harnessing Data Science to Tackle the Challenges of Snow Water Equivalent Prediction</p>
</div>
""", unsafe_allow_html=True)

st.image(header_image, use_container_width=True, caption="Project Overview Image")

# --- Project Overview Section ---
st.markdown("""
<div class="section card">
    <h2>Project Overview</h2>
    <p>
        This project applies advanced machine learning techniques to predict Snow Water Equivalent (SWE), a critical metric 
        for water resource management, flood forecasting, and environmental planning.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Why is SWE Difficult to Predict? Section with Diagram Image ---
st.markdown("""
<div class="section card">
    <h2>Why is SWE Difficult to Predict?</h2>
    <p>
        SWE prediction remains a significant challenge due to:
    </p>
    <ul>
        <li><strong>Meteorological Variability:</strong> Constant changes in weather conditions affect snow accumulation and melting.</li>
        <li><strong>Complex Terrain Dynamics:</strong> Variations in elevation, aspect, and land cover create non-uniform snow distribution.</li>
        <li><strong>Nonlinear Processes:</strong> Conversion of snowfall into meltwater is influenced by multiple, interacting factors.</li>
        <li><strong>Data Limitations:</strong> Inconsistencies and uncertainties in remote sensing and ground-based measurements.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.image(swe_diagram, use_container_width=True, caption="Diagram of SWE Prediction Challenges")

# --- Navigation Instructions ---
st.markdown("""
<div class="section">
    <h2>Navigation</h2>
    <p>
        To explore this project further, please use the sidebar (now on the right) to navigate among the following pages:
    </p>
    <ol>
        <li>Landing Page</li>
        <li>Background</li>
        <li>Less Data</li>
        <li>Mo Data</li>
        <li>Conclusions</li>
    </ol>
</div>
""", unsafe_allow_html=True)
