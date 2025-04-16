import os
import sys
import streamlit as st
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bias_detection_utils import predict_bias, show_bias_visualizations

BIAS_LABELS = [
    "Sentiment Bias",
    "Coded Bias",
    "Sentiment & Coded Bias",
    "Communal Bias",
    "Neutral/Positive"
]

BIAS_DESCRIPTIONS = {
    "Sentiment Bias": "Tone-based bias, such as overly emotional or dismissive feedback.",
    "Coded Bias": "Indirect bias using coded language often influenced by gender stereotypes.",
    "Sentiment & Coded Bias": "Feedback combining both emotional tone and coded language.",
    "Communal Bias": "Expectations based on traditional gender roles, such as being nurturing.",
    "Neutral/Positive": "No detectable gender bias."
}

REQUIRED_COLUMNS = ['feedback_text', 'gender', 'department', 'manager_name']

def run():
    st.title("🔍 Bias Detection in Manager Feedback")
    st.write("This module identifies and summarizes gender-related bias in manager feedback using an AI model.")

  
    uploaded_file = st.file_uploader("📤 Upload Feedback CSV", type="csv", help="Required columns: feedback_text, gender, department, manager_name")

    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
     
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"❌ The following required columns are missing: {', '.join(missing_cols)}")
            return
        else:
            st.success("✅ File uploaded and validated successfully.")
    else:
        st.info("ℹ️ You can upload your own feedback CSV, or the default dataset will be used.")
        try:
            df = pd.read_csv("Feedbacks/testing_bias_dataset_labeled_predictions_1.csv")
        except FileNotFoundError:
            st.warning("⚠️ No default test file found. Please upload a file to begin.")
            return

    if st.button("🧠 Run Bias Detection Model"):
        with st.spinner("Analyzing feedback for gender bias..."):
            labeled_df = predict_bias(df)
            st.session_state['bias_df'] = labeled_df
            st.success("🎉 Bias prediction complete!")
