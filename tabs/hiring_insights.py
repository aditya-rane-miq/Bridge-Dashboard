# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run():
    st.title("Hiring Data Dashboard")

    uploaded_file = st.file_uploader("Upload the Excel File", type=["xlsx"])

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
        all_data = []

        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df['Month'] = sheet  # Add the sheet name as month
            all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)

        # Clean up and format dates
        for col in ['Approved', 'On hold', 'Sourcing start', 'Interview start', 'Interview end']:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        st.subheader("Raw Data Preview")
        st.dataframe(df)

        # GENDER DISTRIBUTION
        st.subheader("Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        st.bar_chart(gender_counts)

        # STAGE-WISE ANALYSIS
        st.subheader("Hiring Stages by Gender")
        stages = ['Approved', 'Sourcing start', 'Interview start', 'Interview end', 'Offered', 'Filled']
        stage_data = {}

        for stage in stages:
            stage_data[stage] = df[df[stage].notna()]['Gender'].value_counts()

        stage_df = pd.DataFrame(stage_data).fillna(0)
        st.bar_chart(stage_df)

        # FUNNEL ANALYSIS
        st.subheader("Hiring Funnel by Gender")
        fig, ax = plt.subplots()
        stage_df.T.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel("Candidate Count")
        ax.set_title("Hiring Funnel Breakdown")
        st.pyplot(fig)

        # APPROVAL RATE BY GENDER
        st.subheader("Approval Rate by Gender")
        total_by_gender = df['Gender'].value_counts()
        approved_by_gender = df[df['Approved'].notna()]['Gender'].value_counts()
        approval_rate = (approved_by_gender / total_by_gender) * 100

        st.bar_chart(approval_rate)

        # FINAL HIRES BY GENDER
        st.subheader("Filled Positions by Gender")
        filled_by_gender = df[df['Filled'].notna()]['Gender'].value_counts()
        st.bar_chart(filled_by_gender)

        # REGION-WISE ANALYSIS
        st.subheader("Gender Distribution by BU Region")
        region_gender = df.groupby(['BU Region', 'Gender']).size().unstack().fillna(0)
        st.dataframe(region_gender)

        st.bar_chart(region_gender)

