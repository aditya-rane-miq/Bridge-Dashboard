import streamlit as st
from Mentorship import load_model, match_employee_with_mentor,match_employee_with_mentee, load_backend_mentor_data,load_backend_data,match_employee_with_training

# Load model and data once
tokenizer, model = load_model()
mentors_df = load_backend_mentor_data()
complete_df=load_backend_data()

# # UI
def run():
    st.set_page_config(page_title="Mentor Matcher", layout="wide")
    st.title("🤝 Smart Mentorship & Reverse Mentorship Matcher")
    # Centered radio button for mentorship type selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mentorship_type = st.radio("Select Mentorship Type", ["Mentorship", "Reverse Mentorship", "Training Material"], horizontal=True)

    # Form layout below the selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.header("Enter Employee Details")
        with st.form(key="employee_form"):

            # # Optional: For Training Material selection, we don't need high potential
            # if mentorship_type in ["Mentorship", "Reverse Mentorship"]:
            #     high_potential = st.selectbox("High Potential", ["Yes", "No"])
            # else:
            #     high_potential = None  # No high potential needed for training material

            # Collect development area and career goals depending on mentorship type
            if mentorship_type == "Mentorship":
                emp_id = st.text_input("Employee ID")
                role_level = st.selectbox("Mentor Role Level", ["Select","Mid", "Senior","Lead"])
                development_area = st.text_input("Preferred Development Areas (e.g., Cloud Computing,SQL,NLP)")
                career_goal = st.text_input("Career Goal (e.g., Data Science,Devops,Analytics )")
            elif mentorship_type == "Reverse Mentorship":
                emp_id = st.text_input("Mentor ID")
                role_level = st.selectbox("Mentee Role Level", ["Select","Entry","Junior","Mid", "Senior","Lead"])
                development_area = st.text_input("Your Expertise (What you can mentor on . e.g., Data science, Analytics)")
                career_goal = st.text_input("Reverse Mentorship Topic (What type of mentee you'd like to help .e.g., NLP, Cloud Computing)")
            else:  # Training Material
                emp_id = st.text_input("Employee ID")
                role_level = st.selectbox("Next Immediate Role Level", ["Select","Junior","Mid", "Senior","Lead"])
                development_area = st.text_input("Preferred Development Areas (e.g., Data Analytics, Leadership)")
                career_goal = st.text_input("Career Goal")

            submitted = st.form_submit_button("Find Matches")

        if submitted:
            employee = {
                "EmpID": emp_id,
                "Role_Level": role_level,
                # "High_Potential_Flag": high_potential,
                "development_area": development_area,
                "Career_Goal": career_goal
            }

            if mentorship_type in ["Mentorship"]:
                matches = match_employee_with_mentor(employee, mentors_df, tokenizer, model, mentorship_type)
                if matches:
                    st.success(f"✅ Top {mentorship_type.lower()} matches:")
                    st.dataframe(matches[:5])
                else:
                    st.warning("🚫 No suitable matches found.")
            if mentorship_type in ["Reverse Mentorship"]:
                matches = match_employee_with_mentee(employee, mentors_df, tokenizer, model, mentorship_type)
                if matches:
                    st.success(f"✅ Top {mentorship_type.lower()} matches:")
                    st.dataframe(matches[:5])
                else:
                    st.warning("🚫 No suitable matches found.")
            
            elif mentorship_type == "Training Material":
                training_matches = match_employee_with_training(employee, complete_df, tokenizer, model)
                if training_matches:
                    st.success("✅ Top Relevant Training Materials:")
                    st.dataframe(training_matches[:5])
                else:
                    st.warning("🚫 No relevant training materials found.")

# if __name__ == "__main__":
#     run()

