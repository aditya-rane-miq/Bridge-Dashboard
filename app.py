import streamlit as st

st.set_page_config(page_title="B.R.I.D.G.E.", layout="wide")

# Add logo and title
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)  # AI logo
with col2:
    st.title("B.R.I.D.G.E. – Bias Removal in Decisions for Gender Equality")

# Define tab titles
tabs = st.tabs([
    "AI-Powered Blind Recruitment",
    "Diverse Candidate Sourcing",
    "Bias Detection in Feedback",
    "Data-Driven Promotions",
    "AI-Powered Salary Audits",
    "Sentiment Analysis in Surveys",
    "Network Analysis",
    "Smart Matching for Mentorship"
])

with tabs[0]:
    st.header("AI-Powered Blind Recruitment")
    st.image("https://cdn-icons-png.flaticon.com/512/4331/4331067.png", width=50)
    st.write("""
        - **Redacts gender-identifying information** from resumes using AI.
        - Applies **NLP to assess skills and experience objectively**.
    """)

with tabs[1]:
    st.header("Diverse Candidate Sourcing")
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094835.png", width=50)
    st.write("""
        - **Analyzes job descriptions** for gender-biased language and suggests improvements.
        - **Targets underrepresented groups** with AI-driven outreach strategies.
    """)

with tabs[2]:
    st.header("Bias Detection in Feedback")
    st.image("https://cdn-icons-png.flaticon.com/512/6565/6565627.png", width=50)
    st.write("""
        - AI **flags subjective or biased language** in feedback.
        - Compares evaluation trends across genders to **highlight disparities**.
    """)

with tabs[3]:
    st.header("Data-Driven Promotions")
    st.image("https://cdn-icons-png.flaticon.com/512/4129/4129623.png", width=50)
    st.write("""
        - Tracks **contributions, leadership, and peer feedback** using data.
        - Identifies promotion-ready talent based on **objective metrics**.
    """)

with tabs[4]:
    st.header("AI-Powered Salary Audits")
    st.image("https://cdn-icons-png.flaticon.com/512/2038/2038854.png", width=50)
    st.write("""
        - **Audits salary data in real-time** to identify pay gaps.
        - Recommends salary corrections using factors like role, performance, and experience.
    """)

with tabs[5]:
    st.header("Sentiment Analysis in Surveys")
    st.image("https://cdn-icons-png.flaticon.com/512/990/990168.png", width=50)
    st.write("""
        - Analyzes internal surveys to find **gender-based sentiment differences**.
        - Provides actionable insights to improve **inclusivity and engagement**.
    """)

with tabs[6]:
    st.header("Network Analysis")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
    st.write("""
        - Maps collaboration and communication patterns.
        - Ensures women are included in **high-impact project opportunities**.
    """)

with tabs[7]:
    st.header("Smart Matching for Mentorship")
    st.image("https://cdn-icons-png.flaticon.com/512/3771/3771602.png", width=50)
    st.write("""
        - AI **pairs mentees with mentors** based on aligned goals and skills.
        - Encourages **reverse mentorship** to foster inclusive leadership.
    """)

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>© 2025 B.R.I.D.G.E. – AI for Gender Equality | Powered by Streamlit</p>
""", unsafe_allow_html=True)