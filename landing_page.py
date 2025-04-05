import streamlit as st

# Streamlit app
st.title('Mo Data Sno Problem')
st.subheader('Problem')

# Problem Description
st.write("International aid often follows political pathways rather than data-driven approaches, resulting in inefficiencies that can lead to missed opportunities in addressing the root causes of terrorism. Factors such as economic instability, lack of education, and governance issues in at-risk regions are often overlooked. To tackle this challenge, a data-driven framework is proposed, incorporating geospatial analysis of terrorism incident data, data modeling of international aid flows, and sentiment analysis on NGO and regional narratives to identify areas of potential impact. Additionally, an AI-driven recommendation system would be developed to optimize funding allocation, ensuring resources are directed to where they are needed most.")


# 2nd Info section
st.subheader('Stuff 2')

# Create two columns
col1, col2 = st.columns(2)

# Add a text input in each column
with col1:
    input1 = st.write("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

with col2:
    input2 = st.write("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

# Addition info Section
st.subheader('Mo Stuff')
st.write("stuffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")

# Group information
st.subheader('Names')
st.write("Names")