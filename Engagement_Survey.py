import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
def run_sentiment_analysis(uploaded_file):
    # Load environment variable for Hugging Face
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Fallback to hardcoded token
    if huggingface_token is None:
        huggingface_token = "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"

    if not huggingface_token:
        raise ValueError("Hugging Face API token not found. Please set the 'HUGGINGFACEHUB_API_TOKEN' environment variable or provide a fallback token.")

    # Load and preprocess data
    df = pd.read_excel(uploaded_file)
    df.fillna(0, inplace=True)

    response_cols = [col for col in df.columns if "Q" in col and "Response" in col]
    df_long = df.melt(
        id_vars=['EmpID', 'Gender', 'Experience', 'Dept', 'EmpType', 'Engagement survey Category'],
        value_vars=response_cols,
        var_name='Question_Num',
        value_name='Response'
    )
    df_long.dropna(subset=['Response'], inplace=True)
    df_long['Response'] = df_long['Response'].astype(int)

    # Sentiment classification function
    def classify_sentiment(score):
        if score >= 4:
            return 'Positive'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Negative'

    df_long['Sentiment'] = df_long['Response'].apply(classify_sentiment)

    # Sentiment and scaling analysis
    sentiment_by_gender = df_long.groupby(['Gender', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_by_gender_percent = sentiment_by_gender.div(sentiment_by_gender.sum(axis=1), axis=0) * 100

    def convert_to_sentiment(score):
        if score <= 2:
            return -1
        elif score == 3:
            return 0
        else:
            return 1

    df['JobSatisfaction Score'] = df[response_cols].mean(axis=1)
    df['Sentiment Score'] = df[response_cols].applymap(convert_to_sentiment).mean(axis=1)

    scaler = MinMaxScaler()
    df['NormSentiment'] = scaler.fit_transform(df[['Sentiment Score']])
    df['NormJobSatisfaction'] = scaler.fit_transform(df[['JobSatisfaction Score']])

    # Inclusion and belonging analysis
    inclusion_keywords = ['inclusion', 'diversity', 'respect', 'belong', 'voice', 'opinions']
    inclusion_mask = df['Engagement survey Category'].str.lower().str.contains('|'.join(inclusion_keywords), na=False)
    df['InclusionBelongingScore'] = df[response_cols].where(inclusion_mask, pd.NA).mean(axis=1)
    df['InclusionBelongingScore_Filled'] = df['InclusionBelongingScore'].fillna(0)
    df['NormInclusion'] = scaler.fit_transform(df[['InclusionBelongingScore_Filled']])

    # Calculate Engagement Heat
    df['Engagement HeatScore'] = (
        df['NormSentiment'] * 0.4 +
        df['NormJobSatisfaction'] * 0.3 +
        df['NormInclusion'] * 0.3
    )

    # Gender sentiment gap calculation
    gender_sentiment_gap = df.groupby("Gender")["Sentiment Score"].mean()
    sentiment_gap_value = gender_sentiment_gap.get("Male", 0) - gender_sentiment_gap.get("Female", 0)

    # Department and Experience analysis
    dept_analysis = df.groupby("Dept")[["JobSatisfaction Score", "Sentiment Score", "Engagement HeatScore"]].mean()
    experience_analysis = df.groupby("Experience")[["JobSatisfaction Score", "Sentiment Score", "Engagement HeatScore"]].mean()
    inclusion_dpt = df.groupby("Dept")[["NormInclusion"]].mean().rename(columns={"NormInclusion": "Inclusion Score"})

    # Summary insights
    total_employees = df['EmpID'].nunique()
    positive_pct = (df["Sentiment Score"] > 0).mean() * 100
    negative_pct = (df["Sentiment Score"] < 0).mean() * 100
    avg_satisfaction = df['JobSatisfaction Score'].mean()
    avg_inclusion = df['InclusionBelongingScore'].mean()
    lowest_dept = df.groupby("Dept")["Engagement HeatScore"].mean().idxmin()
    lowest_dept_score = df.groupby("Dept")["Engagement HeatScore"].mean().min()
    lowest_exp = df.groupby("Experience")["Engagement HeatScore"].mean().idxmin()
    lowest_exp_score = df.groupby("Experience")["Engagement HeatScore"].mean().min()
    
        # Top mentioned themes (Theme Frequency)
    theme_freq = df.groupby("Engagement survey Category")["Sentiment Score"].mean().sort_values(ascending=False)
    top_themes_ = theme_freq[theme_freq > 0]

    top_themes_list = top_themes_.head(3).index.tolist()

    top_themes = top_themes_.head(3)
    bottom_themes = theme_freq.tail(3)

    at_risk_pct = (df["Engagement HeatScore"] < 0.3).mean() * 100

    # Executive summary generation
    insight_summary = f"""
    1. **Total Employees**: {total_employees} employees participated.
    2. **Positive Sentiment**: {positive_pct:.2f}%
    3. **Negative Sentiment**: {negative_pct:.2f}%
    4. **Avg Job Satisfaction**: {avg_satisfaction:.2f}
    5. **Inclusion & Belonging Score**: {avg_inclusion:.2f}
    6. **Sentiment Gap (M - F)**: {sentiment_gap_value:.2f}
    7. **Lowest Dept**: {lowest_dept} ({lowest_dept_score:.2f})
    8. **Experience Group at Risk**: {lowest_exp} ({lowest_exp_score:.2f})
    9. **Top Themes**: {', '.join(top_themes_list)}
    11. **% At-Risk Employees**: {at_risk_pct:.2f}%
    """

        # Prepare the prompt for LLM analysis
# Define the prompt template in Mistral-compatible format
    template = """ Below are the Key Insights and Recommendations based on our engagement survey:
        
    {insights} """

    # Build the LangChain prompt
    prompt = PromptTemplate(template=template, input_variables=["insights"])
    from langchain_core.output_parsers import StrOutputParser

    # Set up the LLM
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        huggingfacehub_api_token=huggingface_token
    )

    # Output parser
    chain = prompt | llm | StrOutputParser()

    # Run the chain with your insights as the question
    result = chain.invoke({"insights": insight_summary})


    # llm = HuggingFaceHub(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #     model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
    #     huggingfacehub_api_token=huggingface_token
    # )

    # chain = LLMChain(llm=llm, prompt=prompt_template)
    # result = chain.run(insights=insight_summary)

        # Aggregated data for easy plotting
    summary_data = {
        "Card Title": [
            "Engagement Heat Score (Avg)",
            "Positive Sentiment %",
            "Negative Sentiment %",
            "Avg Job Satisfaction Index",
            "Inclusion & Belonging Score (Avg)",
            "Sentiment Gap (M - F)",
            "Lowest Scoring Department",
            "Highest Scoring Department",
            "Experience Group at Risk",
            "Top Mentioned Themes",
            "Bottom Mentioned Themes",  # Added bottom themes title
            "% At-Risk Employees (Heat < 0.3)"
        ],
        "Performance": [
            df["Engagement HeatScore"].mean().round(2),
            (df["Sentiment Score"] > 0).mean().round(3) * 100,
            (df["Sentiment Score"] < 0).mean().round(3) * 100,
            df["JobSatisfaction Score"].mean().round(2),
            df["InclusionBelongingScore"].mean().round(2),
            round(sentiment_gap_value, 2),
            dept_analysis["Engagement HeatScore"].idxmin(),
            dept_analysis["Engagement HeatScore"].idxmax(),
            experience_analysis["Engagement HeatScore"].idxmin(),
            ", ".join(top_themes_list),
            ", ".join(bottom_themes.index.tolist()),  # Added bottom themes list
            (df["Engagement HeatScore"] < 0.3).mean().round(2) * 100
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    return {
        "summary_df": summary_df,
        "gender_sentiment_gap": gender_sentiment_gap,
        "dept_analysis": dept_analysis,
        "experience_analysis": experience_analysis,
        "inclusion_dpt":inclusion_dpt,
        "top_themes": top_themes,
        "bottom_themes": bottom_themes,
        "sentiment_gap_value": sentiment_gap_value,
        "result": result
    }
