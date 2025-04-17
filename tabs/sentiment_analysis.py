import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Engagement_Survey import run_sentiment_analysis
import altair as alt

def run():
    # === Streamlit Page Setup ===
# st.set_page_config(page_title="Engagement Dashboard", layout="wide")
    st.markdown("<h1 style='text-align: center;'>📊 Engagement Insights Dashboard</h1>", unsafe_allow_html=True)

    # === Custom CSS to Adjust Sidebar and Layout ===
    st.markdown(
        """
        <style>
        .css-1d391kg {
            width: 50px;
        }
        .metric-card {
            background-color: #F9FAFB;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            text-align: center;
            height: 100%;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-value {
            font-size: 1.1rem;
            font-weight: 500;
            color: #111827;
            overflow-wrap: break-word;
            white-space: normal;
        }
        .stColumns > div {
            flex: 1 1 20%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Upload file on top
    st.header("📁 Upload Your Survey data")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Process the file here (e.g., read Excel into pandas DataFrame)
        st.success("File uploaded successfully!")
        # Add code here to process the file, e.g., pd.read_excel(uploaded_file)


    # === On File Upload ===
    if uploaded_file:
        with st.spinner("Running analysis..."):
            analysis_results = run_sentiment_analysis(uploaded_file)

            summary_df = analysis_results.get("summary_df")
            gender_sentiment_gap_df = analysis_results.get("gender_sentiment_gap")
            dept_analysis_df = analysis_results.get("dept_analysis")
            experience_analysis_df = analysis_results.get("experience_analysis")
            inclusion_analysis_df = analysis_results.get("inclusion_dpt")
            top_themes = analysis_results.get("top_themes")
            bottom_themes = analysis_results.get("bottom_themes")
            sentiment_gap_value = analysis_results.get("sentiment_gap_value")
            llm_insight = analysis_results.get("result")
            

            df_gender_sentiment_gap1 = pd.DataFrame(gender_sentiment_gap_df).reset_index().rename(columns={
                'Gender': 'Gender',
                'Sentiment Score': 'Sentiment Score'
            })

            # Department analysis
            dept_analysis_df1 = pd.DataFrame(dept_analysis_df).reset_index()
            # Experience analysis
            experience_analysis_df1 = pd.DataFrame(experience_analysis_df).reset_index()
            inclusion_analysis_df1 = pd.DataFrame(inclusion_analysis_df).reset_index()

            
            
            required_dfs = [
                summary_df, df_gender_sentiment_gap1, dept_analysis_df1, experience_analysis_df1,inclusion_analysis_df1,
                top_themes, bottom_themes
            ]
            if any(df is None or df.empty for df in required_dfs) or sentiment_gap_value is None or llm_insight is None:
                st.error("Some expected data is missing or empty. Please check the analysis function.")
                st.stop()

        summary_dict = dict(zip(summary_df["Card Title"], summary_df["Performance"]))

        def get_val(key):
            val = summary_dict.get(key, "N/A")
            if isinstance(val, float):
                return round(val, 3)
            return val

        card_style = """
        <style>
        .metric-card {
            background-color: #DBEAFE;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            text-align: center;
            height: 100%;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #6B7280;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-size: 0.9rem;
            font-weight: 500;
            color: #111827;
            overflow-wrap: break-word;
            white-space: normal;
        }
        </style>
        """
        st.markdown(card_style, unsafe_allow_html=True)

        def render_metric(title, value, icon=""):
            display_title = f"{icon} {title}" if icon else title
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{display_title}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("📋 Engagement Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_metric("Engagement Heat Score", get_val("Engagement Heat Score (Avg)"), "👥")
            render_metric("Inclusion & Belonging Score", get_val("Inclusion & Belonging Score (Avg)"), "🌈")
            render_metric("At-Risk Experience", get_val("Experience Group at Risk"), "🦳")

        with col2:
            render_metric("Positive Sentiment %", f"{get_val('Positive Sentiment %')}%", "😊")
            render_metric("% At-Risk Employees", f"{get_val('% At-Risk Employees (Heat < 0.3)')}%", "⚠️")
            top_theme = get_val("Top Mentioned Themes").split(",")[0].strip()
            render_metric("Top Engagement Theme", top_theme, "🏆")

        with col3:
            render_metric("Negative Sentiment %", f"{get_val('Negative Sentiment %')}%", "☹️")
            render_metric("Sentiment Gap (F - M)", get_val("Sentiment Gap (M - F)"), "➖")
            bottom_theme = get_val("Bottom Mentioned Themes").split(",")[0].strip()
            render_metric("Bottom Engagement Theme", bottom_theme, "📉")

        with col4:
            render_metric("Job Satisfaction Score", get_val("Avg Job Satisfaction Index"), "💼")
            render_metric("Lowest Engagement Department", get_val("Lowest Scoring Department"), "🏢")
            render_metric("Highest Engagement Department", get_val("Highest Scoring Department"), "🏢")




        def get_column_names(df):
            if isinstance(df, pd.DataFrame):
                return df.columns.tolist()
            elif isinstance(df, pd.Series):
                return [df.name]
            return ["Not a DataFrame or Series"]
        

        st.markdown("<hr style='border:1px solid #ccc; margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
        st.markdown("## 💡 Key Visual Trends")

        import altair as alt

        # === Chart Helper with Data Labels ===
        def bar_chart_with_labels(df, x_col, y_col, title, color="#4C78A8"):
            base = alt.Chart(df).encode(
                x=alt.X(f"{x_col}:N", title=x_col),
                y=alt.Y(f"{y_col}:Q", title="Sentiment Score"),
                tooltip=[x_col, y_col]
            )

            bars = base.mark_bar(color=color)
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-2,  # slight offset
                fontSize=12
            ).encode(
                text=alt.Text(f"{y_col}:Q", format=".2f")
            )

            return (bars + text).properties(title=title, width="container", height=350)

        # === Layout: 2 charts per row ===
        col1, col2 = st.columns(2)

        # === Gender Sentiment Chart ===
        shared_color_scale = alt.Scale(
        domain=['JobSatisfaction Score', 'Sentiment Score', 'Engagement HeatScore'],
        range=['#4C78A8', '#F58518', '#E45756'])
        shared_color_scale_ = alt.Scale(
        domain=['Inclusion Score'],
        range=['#EECA3B']
    )

    # === Gender Sentiment Chart ===
        with col1:
                st.subheader("📊 Gender Sentiment")
                if isinstance(df_gender_sentiment_gap1, pd.DataFrame) and 'Gender' in df_gender_sentiment_gap1.columns and 'Sentiment Score' in df_gender_sentiment_gap1.columns:
                    chart = alt.Chart(df_gender_sentiment_gap1).mark_bar(size=55).encode(
                        x=alt.X('Gender:N', title='Gender',axis=alt.Axis(labelAngle=-45,domain=True)),
                        y=alt.Y('Sentiment Score:Q', title='Sentiment Score'),
                        color=alt.value('#4C78A8'),
                        tooltip=['Gender:N', 'Sentiment Score:Q']
                    ).properties(
                        height=400
                    ).configure_view(
                        stroke=None  # Remove outer border
                    ).configure_axis(
                        grid=False  # Remove background grid lines
                    ).configure_title(
                                fontSize=6,  # 👈 Adjust title font size here
                                anchor='start'
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error("Gender or Sentiment Score column missing.")

        # === Department Sentiment Chart ===
        with col2:
            st.subheader("📊 Department Score")
            if isinstance(dept_analysis_df1, pd.DataFrame) and \
                'Dept' in dept_analysis_df1.columns and \
                all(col in dept_analysis_df1.columns for col in ['JobSatisfaction Score', 'Sentiment Score', 'Engagement HeatScore']):

                dept_df = dept_analysis_df1.melt(
                    id_vars=['Dept'],
                    value_vars=['JobSatisfaction Score', 'Sentiment Score', 'Engagement HeatScore'],
                    var_name='Metric',
                    value_name='Value'
                )
                

                chart = alt.Chart(dept_df).mark_bar().encode(
                    x=alt.X('Dept:N', title='Department', axis=alt.Axis(labelAngle=-45,domain=True)),
                    xOffset='Metric:N',
                    y=alt.Y('Value:Q', title='Score'),
                    color=alt.Color('Metric:N', scale=shared_color_scale, legend=alt.Legend(orient='top', labelFontSize=11,title=None)),
                    tooltip=['Dept:N', 'Metric:N', 'Value:Q']
                ).properties(
                    height=400
                ).configure_view(
                    stroke=None
                ).configure_axis(
                    grid=False
                ).configure_title(
                fontSize=6,  # 👈 Adjust title font size here
                anchor='start'
            )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.error("Required columns are missing in the dataset.")

        # === Second Row: Experience Sentiment Chart (full-width) ===
        col3,col4 = st.columns(2)

        with col3:
            st.subheader("📊 Experience Group")
            
            if isinstance(experience_analysis_df1, pd.DataFrame) and \
                'Experience' in experience_analysis_df1.columns and \
                all(col in experience_analysis_df1.columns for col in ['JobSatisfaction Score', 'Sentiment Score', 'Engagement HeatScore']):
        # Small dropdown styling
                st.markdown("""
                    <style>
                    div[data-baseweb="select"] .select__menu {
                    font-size: 9px !important;  /* Adjust the font size as needed */
                    }        
                    </style>
                """, unsafe_allow_html=True)
                # Dropdown to select metric
                selected_metric = st.selectbox(
                    "Select Metric",
                    options=['JobSatisfaction Score', 'Sentiment Score', 'Engagement HeatScore'],
                    index=0,
                    key='metric_selectbox'
                )
                # Prepare data
                exp_df = experience_analysis_df1[['Experience', selected_metric]].copy()
                exp_df = exp_df.rename(columns={selected_metric: 'Value'})
                exp_df['Metric'] = selected_metric

                # Create bar chart
                chart = alt.Chart(exp_df).mark_bar().encode(
                    x=alt.X('Experience:N', title='Experience Level', axis=alt.Axis(labelAngle=-45,domain=True)),
                    y=alt.Y('Value:Q', title='Score'),
                    color=alt.Color('Metric:N', scale=shared_color_scale,legend=None),
                    tooltip=['Experience:N', 'Metric:N', 'Value:Q']
                ).properties(
                    height=350
                ).configure_view(
                    stroke=None
                ).configure_axis(
                    grid=False
                )

                st.altair_chart(chart, use_container_width=True)

            else:
                st.error("Required columns are missing in the dataset.")

        with col4:
            st.subheader("📊 Inclusion & Diversity")
            
            if isinstance(inclusion_analysis_df1, pd.DataFrame) and \
                'Dept' in inclusion_analysis_df1.columns and 'Inclusion Score' in inclusion_analysis_df1.columns:
                st.markdown("<div style='height: 95px;'></div>", unsafe_allow_html=True)

                # Add non-null label for color
                inclusion_analysis_df1['Color Label'] = 'Inclusion Score'

                chart = alt.Chart(inclusion_analysis_df1).mark_bar(size=55).encode(
                    x=alt.X('Dept:N', title='Department', axis=alt.Axis(labelAngle=-45, domain=True)),
                    y=alt.Y('Inclusion Score:Q', title='Inclusion Score'),
                    color=alt.Color('Color Label:N',
                                    scale=alt.Scale(range=['#EECA3B']),
                                    legend=alt.Legend(
                                        orient='top',
                                        direction='horizontal',title=None
                                    )),
                    tooltip=['Dept:N', 'Inclusion Score:Q']

                ).properties(
                    height=350
                ).configure_view(
                    stroke=None
                ).configure_axis(
                    grid=False
                )

                st.altair_chart(chart, use_container_width=True)

            else:
                st.error("Required columns are missing in the dataset.")

        col5, col6 = st.columns(2)
        with col5:
            st.subheader("📋 Top Survey Themes")
            st.write(top_themes)
        with col6:
            st.subheader("📋 Bottom Survey Themes")
            st.write(bottom_themes)

        # === LLM Insight ===
        st.markdown("<hr style='border:1px solid #ccc; margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
        st.markdown("## 🧠 AI-Powered Action Centre")
        st.info(llm_insight)
