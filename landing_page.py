import streamlit as st
from streamlit_lottie import st_lottie
import json

st.set_page_config(layout="wide")
st.title("❄️ Mo Data Sno Problem ❄️")

# Load the Lottie animation from local file
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Path to your local Lottie JSON file
lottie_snow = load_lottie_file('animations/snowfall.json')

# Streamlit app content

st.subheader('Problem Statement')

# Problem Description
st.write("""
International aid often follows political pathways rather than data-driven approaches, resulting in inefficiencies that can lead to missed opportunities in addressing the root causes of terrorism. 
Factors such as economic instability, lack of education, and governance issues in at-risk regions are often overlooked. 
To tackle this challenge, a data-driven framework is proposed, incorporating geospatial analysis of terrorism incident data, 
data modeling of international aid flows, and sentiment analysis on NGO and regional narratives to identify areas of potential impact. 
Additionally, an AI-driven recommendation system would be developed to optimize funding allocation, 
ensuring resources are directed to where they are needed most.
""")

# Display the Lottie animation in two columns
col1, col2 = st.columns(2)

with col1:
    st_lottie(lottie_snow, speed=1, loop=True, quality="high", height=400)

with col2:
    st_lottie(lottie_snow, speed=1, loop=True, quality="high", height=400)
# 2nd Info section
st.subheader('Impact / Market Opportunity')

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.write("""
Declining snowpack levels due to climate change are creating cascading challenges across critical sectors, threatening regional economies and highlighting the need for innovative solutions. 
The U.S. winter recreation industry, valued at 70 billion annually, faces shorter seasons and reduced snowfall, with snow-related visitation projected to drop 40–60 percent by century’s end. 
This could translate into annual economic losses between 1.23 billion and 2.05 billion (Scott et al., 2023). 
Agriculture and energy sectors are also at risk. In Washington’s Yakima Basin, decreased snowpack may reduce stream flows, causing 
13–70 million in annual losses by mid-century (US EPA, 2023). 
Hydropower generation in the Western U.S. dropped 11% during the 2022–2023 water year—about 141.6 million MWh of clean energy lost (EIA, 2024).
""")

with col2:
    st.write("""
Accurate snowpack monitoring is essential. Programs like California’s 2022 aerial snow surveys (31 flights at 9.5M) and Colorado’s Gunnison Basin pilot (1M/year) show costs can exceed 300K per flight. 
As climate risks grow, so will demand for monitoring investments. Water from Western U.S. snowpack supports 40 million people and a 1.4 trillion economy (Colorado River). 
Executive Order 14008 underscores the federal priority for climate resilience. 
Intended users include agencies like the U.S. Bureau of Reclamation (BOR), Army Corps of Engineers, and California DWR, which manage critical reservoirs. 
Over the past two years, BOR has awarded ~15M through its Snow Water Supply Forecasting Program, signaling strong market demand for improved snowpack data and tools to optimize water management and protect infrastructure.
""")

# Display the Lottie animation in two columns
col1, col2 = st.columns(2)

with col1:
    st_lottie(lottie_snow, speed=1, loop=True, quality="high", height=400)

with col2:
    st_lottie(lottie_snow, speed=1, loop=True, quality="high", height=400)
    
# Additional info Section
st.subheader('Mo Stuff - The Difference')
st.write("""
The key differentiation of the MVP lies in its integration of advanced machine learning models for real-time, scalable, and cost-effective snowpack monitoring and prediction. 
Unlike existing solutions, which suffer from issues like forcing bias, satellite latency, or operational complexity, the MVP combines basin-wide SWE predictions, distributed SWE mapping, and SnowModel emulation into a single platform. 
It minimizes bias and improves accuracy through physics-informed and graph-based models, achieving lower RMSE and higher R² values compared to alternatives. 
The MVP also reduces reliance on costly lidar surveys by providing comparable insights using existing datasets, 
while its intuitive user interface ensures accessibility for non-technical users like water resource managers. 
Designed for flexibility, it supports scenario testing, delivers actionable insights through interactive dashboards, and scales efficiently to accommodate diverse basins and data sources, 
which makes it a unique and user-centric solution in the snowpack monitoring landscape.
""")

# Group information
st.subheader('Group Member LinkedIn')
st.markdown("[Paulina Alvarado-Goldman](https://www.linkedin.com/in/paulina-alvarado-goldman/), [Branndon Marion](https://www.linkedin.com/in/branndon-marion/), [Philip Monaco](https://www.linkedin.com/in/philmonaco/), [Ross Mower](https://www.linkedin.com/in/rossmower/), [Vincent Qu](https://www.linkedin.com/in/vincentqu/)")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")





