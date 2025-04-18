# B.R.I.D.G.E. – Bias Removal in Decisions for Gender Equality

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/github/license/yourusername/bridge)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
![AI for Good](https://img.shields.io/badge/AI%20for-Gender%20Equality-ff69b4)

**B.R.I.D.G.E.** is an AI-powered framework that removes gender bias from hiring, promotions, and workplace decision-making. By combining NLP, data science, and organizational analytics, B.R.I.D.G.E. helps companies build equitable and inclusive environments—by design, not exception.

---

## 🚀 Key Use Cases

### 🧑‍💼 Recruitment & Hiring
- **Blind Resume Screening:** Redacts gender-identifying details and evaluates based on skills and experience.
- **Bias-Free Job Descriptions:** Flags exclusionary language and suggests neutral alternatives.
- **Diverse Candidate Outreach:** Recommends job postings to underrepresented gender groups using AI-driven matching.

### 🏢 Workplace Culture & Growth
- **Sentiment Analysis:** This tool analyzes employee engagement survey data to provide insights into sentiment, job satisfaction, Engagement Heatscore and inclusion. It highlights performance gaps across demographics, identifies at-risk groups, and surfaces key engagement themes. An AI-generated executive summary, powered by a Large Language Model (LLM), offers actionable recommendations for leadership to improve engagement.
- **Collaboration Network Mapping:** Ensures equal access to high-visibility, high-impact projects.
- **Smart Mentorship Matching:** Connects individuals with mentors based on skills, goals, and experience—regardless of gender. Suggests reverse mentoring opportunities and personalized training resources to support growth in key development areas.

---

## ⚙️ Tech Stack

- **Languages:** Python  
- **Libraries:** spaCy, Transformers, Scikit-learn, Pandas, NetworkX, Matplotlib  
- **NLP Models:** BERT, RoBERTa (via HuggingFace), Mistral-7B-Instruct-v0.3, all-MiniLM-L6-v2
- **Deployment (optional):** Streamlit / Flask  
- **Data Sources:** Resume sets, employee reviews, survey results (mock or real) , HR Data (mock or real) , LMS Data (mock or real), Performance data from Yearly Goals/PI review (mock or real),Individual development plan of employee data (mock or real)


---
## 📋 Instruction to Run the Application

- **Run Application on Local machine** -> 
  You can clone this in your local machine using this in your terminal -
  - git clone https://github.com/aditya-rane-miq/Bridge-Dashboard.git
  - cd Bridge-Dashboard
  - create virtual venv
  - pip install -r requirements.txt
  - streamlit run app.py
- **Recruitment & Hiring -> Hiring Insights:** Manually upload the file 'All Months Hiring Data.xlsx' to run the analysis .
- **Workplace culture -> Sentiment Analysis in Survey :** Manually Upload the file 'Engagement_survey_Raw_data.xlsx' provided under Bridge-Dashboard folder to run the tool  
- **Workplace culture -> Smart Matching for Mentorship :** The 'Inclusive_Mentorship_Program_Raw_data.xlsx' file from the Bridge-Dashboard folder is automatically read by the tool, so no manual upload is necessary. The tool will prompt the user for certain inputs via the UI.



## 🌍 UI Integration

![image](https://github.com/user-attachments/assets/e3c7bf18-c08b-4f5f-b3fd-493221e76218)



