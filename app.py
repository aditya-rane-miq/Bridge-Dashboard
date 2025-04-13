import streamlit as st
import os
import shutil
import uuid
from bias_removal import BlindRecruitmentProcessor
from io import BytesIO
import json

st.set_page_config(page_title="B.R.I.D.G.E.", layout="wide")

# Add logo and title
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)
with col2:
    st.title("B.R.I.D.G.E. – Bias Removal in Decisions for Gender Equality")

# Directories
RESUME_FOLDER = "Input Resumes"
SELECTED_FOLDER = "Selected Resumes"
REJECTED_FOLDER = "Rejected Resumes"

os.makedirs(RESUME_FOLDER, exist_ok=True)
os.makedirs(SELECTED_FOLDER, exist_ok=True)
os.makedirs(REJECTED_FOLDER, exist_ok=True)

# Helper: Get files and assign random candidate IDs
resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]
id_map = {f"Candidate-{uuid.uuid4().hex[:6].upper()}": f for f in resume_files}

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
    st.markdown("Select a resume from the local folder. AI will redact bias terms for a fairer review.")

    # Step 1: Load resumes and map to anonymized IDs
    resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]
    file_id_map = {f"Candidate {i+1}": f for i, f in enumerate(resume_files)}

    # if file_id_map:
    #     selected_id = st.selectbox("Select a Candidate", list(file_id_map.keys()))
        
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    selected_id = st.selectbox("Select a Candidate", list(file_id_map.keys()), key="candidate_select")

    if st.button("Select Candidate"):
        st.session_state.selected_id = selected_id

    if st.session_state.selected_id:
        selected_file_path = os.path.join(RESUME_FOLDER, file_id_map[st.session_state.selected_id])
        with open(selected_file_path, "rb") as f:
            uploaded_file = BytesIO(f.read())

        processor = BlindRecruitmentProcessor()
        with st.spinner("Analyzing resume using AI..."):
            text, redacted_pdf, bias_terms, summary_info = processor.process_resume(uploaded_file)

        # st.write(summary_info)

        # Tabular Info
        st.subheader("📋 Resume Summary")

        if isinstance(summary_info, dict) and summary_info:
            info_keys = [
                "Summary",
                "Suitable Roles",
                "Education",
                "Experience",
                "Projects",
                "Skills",
                "Certifications"
            ]

            data = []
            for key in info_keys:
                value = summary_info.get(key)

                # Skip empty values
                if not value or value == [] or value == {} or value == "":
                    continue

                if isinstance(value, list):
                    value = "<ul>" + "".join(f"<li>{str(v)}</li>" for v in value) + "</ul>"
                elif isinstance(value, dict):
                    value = "<br>".join(f"{k}: {v}" for k, v in value.items())

                data.append((key, value))


            st.markdown("""
                <style>
                .summary-table {
                    border-collapse: collapse;
                    width: 100%;
                    background-color: #000;
                    color: #fff;
                    font-family: sans-serif;
                }
                .summary-table td {
                    border: 1px solid #444;
                    padding: 10px;
                    vertical-align: top;
                }
                .summary-table tr:nth-child(even) {
                    background-color: #111;
                }
                .summary-table tr:hover {
                    background-color: #222;
                }
                .summary-table td ul {
                    margin: 0;
                    padding-left: 20px;
                }
                </style>
            """, unsafe_allow_html=True)


            table_html = "<table class='summary-table'>"
            for label, val in data:
                table_html += f"<tr><td><strong>{label}</strong></td><td>{val}</td></tr>"
            table_html += "</table>"

            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("Could not extract summary from the resume.")



        st.success("Resume processed successfully!")

        if bias_terms:
            st.warning("Bias terms detected.")
        else:
            st.warning("No bias terms detected.")

        st.download_button(
            label="⬇️ Download Redacted PDF",
            data=redacted_pdf,
            file_name="redacted_resume.pdf",
            mime="application/pdf"
        )

        decision = st.radio("Decision:", ["Select", "Reject"])
        if st.button("Submit Decision"):
            destination_folder = SELECTED_FOLDER if decision == "Select" else REJECTED_FOLDER
            os.makedirs(destination_folder, exist_ok=True)

            original_filename = file_id_map[st.session_state.selected_id]
            source_path = os.path.join(RESUME_FOLDER, original_filename)
            destination_path = os.path.join(destination_folder, original_filename)

            os.rename(source_path, destination_path)
            st.success(f"Candidate moved to `{decision}` folder.")
            st.info(f"Original File Name: **{original_filename}**")
            st.subheader("Bias terms detected were: ")
            st.write(bias_terms)
            
            # Optionally clear the session state to refresh
            st.session_state.selected_id = None



# Other tabs (static info for now)
with tabs[1]:
    st.header("Diverse Candidate Sourcing")
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094835.png", width=50)
    st.write("""
        - Analyzes job descriptions for gender-biased language and suggests improvements.
        - Targets underrepresented groups with AI-driven outreach strategies.
    """)

with tabs[2]:
    st.header("Bias Detection in Feedback")
    st.image("https://cdn-icons-png.flaticon.com/512/6565/6565627.png", width=50)
    st.write("""
        - AI flags subjective or biased language in feedback.
        - Compares evaluation trends across genders to highlight disparities.
    """)

with tabs[3]:
    st.header("Data-Driven Promotions")
    st.image("https://cdn-icons-png.flaticon.com/512/4129/4129623.png", width=50)
    st.write("""
        - Tracks contributions, leadership, and peer feedback using data.
        - Identifies promotion-ready talent based on objective metrics.
    """)

with tabs[4]:
    st.header("AI-Powered Salary Audits")
    st.image("https://cdn-icons-png.flaticon.com/512/2038/2038854.png", width=50)
    st.write("""
        - Audits salary data in real-time to identify pay gaps.
        - Recommends salary corrections using factors like role, performance, and experience.
    """)

with tabs[5]:
    st.header("Sentiment Analysis in Surveys")
    st.image("https://cdn-icons-png.flaticon.com/512/990/990168.png", width=50)
    st.write("""
        - Analyzes internal surveys to find gender-based sentiment differences.
        - Provides actionable insights to improve inclusivity and engagement.
    """)

with tabs[6]:
    st.header("Network Analysis")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
    st.write("""
        - Maps collaboration and communication patterns.
        - Ensures women are included in high-impact project opportunities.
    """)

with tabs[7]:
    st.header("Smart Matching for Mentorship")
    st.image("https://cdn-icons-png.flaticon.com/512/3771/3771602.png", width=50)
    st.write("""
        - AI pairs mentees with mentors based on aligned goals and skills.
        - Encourages reverse mentorship to foster inclusive leadership.
    """)

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>© 2025 B.R.I.D.G.E. – AI for Gender Equality | Powered by Streamlit</p>
""", unsafe_allow_html=True)