import streamlit as st
import os
from tabs import (
    blind_recruitment, diverse_sourcing, bias_detection_feedback,
    data_driven_promotions, salary_audits, sentiment_analysis,
    network_analysis, smart_matching, hiring_insights
)

st.set_page_config(page_title="B.R.I.D.G.E.", layout="wide")

# Styling
import streamlit as st

# Load and apply CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)
    st.markdown("### **B.R.I.D.G.E. - Bias Removal in Decisions for Gender Equality**")

st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Choose a Section:", [
    "Recruitment & Hiring",
    "Manager Feedback & Promotions",
    "Workplace Culture"
])

tab_options = {
    "Recruitment & Hiring": {
        "AI-Powered Blind Recruitment": blind_recruitment,
        "Diverse Candidate Sourcing": diverse_sourcing,
        "Hiring Insights": hiring_insights
    },
    "Manager Feedback & Promotions": {
        "Bias Detection in Feedback": bias_detection_feedback,
        "Data-Driven Promotions": data_driven_promotions,
        "AI-Powered Salary Audits": salary_audits
    },
        "Workplace Culture": {
        "Sentiment Analysis in Surveys": sentiment_analysis,
        "Network Analysis": network_analysis,
        "Smart Matching for Mentorship": smart_matching
    }
}

tab_names = list(tab_options[section].keys())
selected_tab = st.sidebar.selectbox("Choose a Tab:", tab_names)

# Run selected tab
tab_module = tab_options[section][selected_tab]
tab_module.run()

# Footer
st.markdown("""
    <hr style="border: 1px solid #333;">
    <p style='text-align: center; color: gray;'>© 2025 B.R.I.D.G.E. – AI for Gender Equality | Powered by Streamlit</p>
""", unsafe_allow_html=True)
