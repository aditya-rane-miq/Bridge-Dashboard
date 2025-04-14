import streamlit as st
import os
import shutil
import uuid
from bias_removal import BlindRecruitmentProcessor
from io import BytesIO
import json

st.set_page_config(page_title="B.R.I.D.G.E.", layout="wide")

# Global Style
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3, h4, h5 {
        color: #ffcc70;
    }
    .sidebar .sidebar-content {
        background-color: #1c1f26;
    }
    .summary-table {
        border-collapse: collapse;
        width: 100%;
        background-color: #000;
        color: #fff;
        font-family: 'Segoe UI', sans-serif;
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
    .reportview-container .main footer {
        visibility: hidden;
    }
    .stDownloadButton>button {
        background-color: #ffcc70;
        color: black;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ffcc70;
        color: black;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)
    st.markdown("### **B.R.I.D.G.E. - Bias Removal in Decisions for Gender Equality**")


# Folder structure
RESUME_FOLDER = "Input Resumes"
SELECTED_FOLDER = "Selected Resumes"
REJECTED_FOLDER = "Rejected Resumes"

os.makedirs(RESUME_FOLDER, exist_ok=True)
os.makedirs(SELECTED_FOLDER, exist_ok=True)
os.makedirs(REJECTED_FOLDER, exist_ok=True)

# Sidebar navigation
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Choose a Section:", [
    "Recruitment & Hiring",
    "Manager Feedback & Promotions",
    "Workplace Culture"
])

tab_options = {
    "Recruitment & Hiring": {
        "AI-Powered Blind Recruitment": 0,
        "Diverse Candidate Sourcing": 1
    },
    "Manager Feedback & Promotions": {
        "Bias Detection in Feedback": 2,
        "Data-Driven Promotions": 3,
        "AI-Powered Salary Audits": 4
    },
    "Workplace Culture": {
        "Sentiment Analysis in Surveys": 5,
        "Network Analysis": 6,
        "Smart Matching for Mentorship": 7
    }
}

tab_names = list(tab_options[section].keys())
selected_tab = st.sidebar.selectbox("Choose a Tab:", tab_names)

# Tab logic
tab_index = tab_options[section][selected_tab]

# Individual tabs
if tab_index == 0:
    st.header("🤖 AI-Powered Blind Recruitment")
    st.markdown("**Upload and analyze resumes with AI to remove gender bias for fairer hiring decisions.**")

    resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]
    file_id_map = {f"Candidate {i+1}": f for i, f in enumerate(resume_files)}

    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    selected_id = st.selectbox("👤 Select a Candidate", list(file_id_map.keys()), key="candidate_select")

    if st.button("🔍 Analyze Resume"):
        st.session_state.selected_id = selected_id

    if st.session_state.selected_id:
        selected_file_path = os.path.join(RESUME_FOLDER, file_id_map[st.session_state.selected_id])
        with open(selected_file_path, "rb") as f:
            uploaded_file = BytesIO(f.read())

        processor = BlindRecruitmentProcessor()
        with st.spinner("Analyzing resume using AI..."):
            text, redacted_pdf, bias_terms, summary_info = processor.process_resume(uploaded_file)

        st.subheader("📋 Resume Summary")
        if isinstance(summary_info, dict) and summary_info:
            info_keys = ["Summary", "Suitable Roles", "Education", "Experience", "Projects", "Skills", "Certifications"]
            data = []
            for key in info_keys:
                value = summary_info.get(key)
                if not value or value in [[], {}, ""]:
                    continue
                if isinstance(value, list):
                    value = "<ul>" + "".join(f"<li>{str(v)}</li>" for v in value) + "</ul>"
                elif isinstance(value, dict):
                    value = "<br>".join(f"{k}: {v}" for k, v in value.items())
                data.append((key, value))

            table_html = "<table class='summary-table'>" + "".join(
                f"<tr><td><strong>{label}</strong></td><td>{val}</td></tr>" for label, val in data
            ) + "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Could not extract summary from the resume.")

        st.success("✅ Resume processed successfully!")

        if bias_terms:
            st.warning("⚠️ Bias terms detected.")
        else:
            st.success("🎉 No bias terms detected!")

        st.download_button(
            label="⬇️ Download Redacted Resume",
            data=redacted_pdf,
            file_name="redacted_resume.pdf",
            mime="application/pdf"
        )

        decision = st.radio("📤 Final Decision:", ["Select", "Reject"])
        if st.button("Submit Decision"):
            destination_folder = SELECTED_FOLDER if decision == "Select" else REJECTED_FOLDER
            os.makedirs(destination_folder, exist_ok=True)
            original_filename = file_id_map[st.session_state.selected_id]
            source_path = os.path.join(RESUME_FOLDER, original_filename)
            destination_path = os.path.join(destination_folder, original_filename)
            os.rename(source_path, destination_path)
            st.success(f"📁 Candidate moved to `{decision}` folder.")
            st.info(f"Original File Name: **{original_filename}**")
            st.subheader("🛑 Bias terms detected were:")
            st.write(bias_terms)
            st.session_state.selected_id = None

elif tab_index == 1:
    st.header("🌍 Diverse Candidate Sourcing")
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094835.png", width=50)
    st.markdown("""
        - Analyze job descriptions to eliminate bias.
        - Use AI to engage diverse and underrepresented groups.
    """)

elif tab_index == 2:
    st.header("📝 Bias Detection in Feedback")
    st.image("https://cdn-icons-png.flaticon.com/512/6565/6565627.png", width=50)
    st.markdown("""
        - Automatically detect biased language in reviews.
        - Compare feedback trends by gender.
    """)

elif tab_index == 3:
    st.header("📈 Data-Driven Promotions")
    st.image("https://cdn-icons-png.flaticon.com/512/4129/4129623.png", width=50)
    st.markdown("""
        - Highlight top performers using objective KPIs.
        - Empower fair career advancement.
    """)

elif tab_index == 4:
    st.header("💰 AI-Powered Salary Audits")
    st.image("https://cdn-icons-png.flaticon.com/512/2038/2038854.png", width=50)
    st.markdown("""
        - Audit compensation for potential gender gaps.
        - Suggest equitable adjustments in real-time.
    """)

elif tab_index == 5:
    st.header("💬 Sentiment Analysis in Surveys")
    st.image("https://cdn-icons-png.flaticon.com/512/990/990168.png", width=50)
    st.markdown("""
        - Detect sentiment differences by gender.
        - Improve organizational inclusivity.
    """)

elif tab_index == 6:
    st.header("🔗 Network Analysis")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
    st.markdown("""
        - Track collaboration patterns.
        - Ensure women have access to high-impact projects.
    """)

elif tab_index == 7:
    st.header("🤝 Smart Matching for Mentorship")
    st.image("https://cdn-icons-png.flaticon.com/512/3771/3771602.png", width=50)
    st.markdown("""
        - Match mentors and mentees based on shared values.
        - Foster leadership through reverse mentoring.
    """)

# Footer
st.markdown("""
    <hr style="border: 1px solid #333;">
    <p style='text-align: center; color: gray;'>© 2025 B.R.I.D.G.E. – AI for Gender Equality | Powered by Streamlit</p>
""", unsafe_allow_html=True)
