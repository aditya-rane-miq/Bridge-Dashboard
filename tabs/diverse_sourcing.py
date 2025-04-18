import streamlit as st
from unbiased_jd_writer import UnbiasedJDWriter

def run():
    st.header("🌍 Diverse Candidate Sourcing")
    st.markdown("""
        - Analyze job descriptions to eliminate bias.
        - Use AI to engage diverse and underrepresented groups.
    """)
    
    st.subheader("✍️ Generate Unbiased Job Description")

    with st.form("jd_form"):
        job_title = st.text_input("Job Title")
        experience = st.text_input("Experience Required (e.g. 3+ years)")
        skills = st.text_area("Key Skills (comma-separated)")
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Internship", "Temporary"])
        location = st.text_input("Location")
        responsibilities = st.text_area("Key Responsibilities")
        preferred_qualifications = st.text_area("Preferred Qualifications")
        company_info = st.text_area("Company Info / Culture")

        submitted = st.form_submit_button("Generate JD")

    if submitted:
        jd_writer = UnbiasedJDWriter()
        input_data = {
            "job_title": job_title,
            "experience": experience,
            "skills": skills,
            "employment_type": employment_type,
            "location": location,
            "responsibilities": responsibilities,
            "preferred_qualifications": preferred_qualifications,
            "company_info": company_info
        }

        with st.spinner("Generating unbiased job description..."):
            jd = jd_writer.create_unbiased_jd(input_data)

        st.subheader("📄 Generated Unbiased Job Description")
        st.code(jd, language='markdown')

        # Bias detection section
        st.subheader("🔍 Bias Analysis in Company Culture")

        with st.spinner("Analyzing company info for biased language..."):
            rule_based_flags = jd_writer.detect_biased_language(company_info)
            ai_feedback = jd_writer.ai_detect_biased_language(company_info)

        if rule_based_flags:
            st.markdown("### 🚨 Rule-Based Bias Flags Detected")
            for category, terms in rule_based_flags.items():
                st.write(f"**{category.replace('_', ' ').title()}**: {', '.join(terms)}")
        else:
            st.success("✅ No common bias patterns found using rule-based detection.")

        st.markdown("### 🤖 AI-Powered Feedback")
        st.info(ai_feedback)
