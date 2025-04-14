import streamlit as st
import os
from io import BytesIO
from bias_removal import BlindRecruitmentProcessor

def run():
    st.header("🤖 AI-Powered Blind Recruitment")
    st.markdown("**Upload and analyze resumes with AI to remove gender bias for fairer hiring decisions.**")

    RESUME_FOLDER = "Input Resumes"
    SELECTED_FOLDER = "Selected Resumes"
    REJECTED_FOLDER = "Rejected Resumes"

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
                if not value:
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
