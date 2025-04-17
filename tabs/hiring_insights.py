# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run():
    st.title("📊 Hiring Data Dashboard")

    uploaded_file = st.file_uploader("📥 Upload the Excel File", type=["xlsx"])

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
        all_data = []

        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df['Month'] = sheet  # Add sheet name as month
            all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)

        # Convert date columns
        for col in ['Approved', 'On hold', 'Sourcing start', 'Interview start', 'Interview end']:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Split screen into 2 columns
        col1, col2 = st.columns(2)

        # Gender Distribution
        with col1:
            st.subheader("👤 Gender Distribution")
            gender_counts = df['Gender'].value_counts()
            fig1, ax1 = plt.subplots()
            sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="Set2", ax=ax1)
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        # Approval Rate by Gender
        with col2:
            st.subheader("✅ Approval Rate by Gender")
            total_by_gender = df['Gender'].value_counts()
            approved_by_gender = df[df['Status'] == 'Approved']['Gender'].value_counts()
            approval_rate = (approved_by_gender / total_by_gender) * 100
            fig2, ax2 = plt.subplots()
            sns.barplot(x=approval_rate.index, y=approval_rate.values, palette="Set3", ax=ax2)
            ax2.set_ylabel("Approval Rate (%)")
            st.pyplot(fig2)

        # Final Hires by Gender
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🏁 Filled Positions by Gender")
            filled_by_gender = df[df['Filled'].notna()]['Gender'].value_counts()
            fig3, ax3 = plt.subplots()
            sns.barplot(x=filled_by_gender.index, y=filled_by_gender.values, palette="coolwarm", ax=ax3)
            ax3.set_ylabel("Count")
            st.pyplot(fig3)

        # Region-wise Gender Distribution
        with col4:
            st.subheader("🌍 Gender by BU Region")
            region_gender = df.groupby(['BU Region', 'Gender']).size().unstack().fillna(0)
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            region_gender.plot(kind='bar', stacked=True, ax=ax4, colormap="Accent")
            ax4.set_ylabel("Count")
            st.pyplot(fig4)

        # Monthly Approval by Gender
        st.subheader("📅 Monthly Approved Count by Gender")
        monthly_approved = df[df['Status'] == 'Approved'].groupby(['Month', 'Gender']).size().reset_index(name='Approved Count')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=monthly_approved, x='Month', y='Approved Count', hue='Gender', ax=ax5, palette="viridis")
        ax5.set_title("Approved Count by Gender per Month")
        ax5.set_ylabel("Approved Count")
        ax5.set_xlabel("Month")
        st.pyplot(fig5)
