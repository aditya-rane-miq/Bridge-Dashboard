import os
from typing import Dict, List
import openai
import re

client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class UnbiasedJDWriter:
    def __init__(self):
        self.model = "mistralai/mistral-7b-instruct"
        self.bias_indicators = {
            "gendered_terms": ["rockstar", "ninja", "dominant", "aggressive", "manpower", "he", "she"],
            "age_bias": ["young", "energetic", "digital native", "recent graduate"],
            "ableist_terms": ["fast-paced", "work hard/play hard", "able-bodied"],
            "exclusionary_phrases": ["must have", "native speaker", "fit in", "aggressively", "thick skin"],
            "cultural_bias": ["beer fridge", "ping pong", "work hard play hard", "bro culture"]
        }

    def generate_prompt(self, inputs: Dict[str, str]) -> str:
        return f"""
You are an expert diversity and inclusion recruiter assistant. Based on the following recruiter input, write a **neutral and inclusive job description** that avoids any form of gender, race, age, or ability bias. Focus on clarity, inclusiveness, and accessibility.

Here is the information:
- **Job Title**: {inputs.get('job_title', '')}
- **Experience Required**: {inputs.get('experience', '')}
- **Key Skills**: {inputs.get('skills', '')}
- **Employment Type**: {inputs.get('employment_type', '')}
- **Location**: {inputs.get('location', '')}
- **Responsibilities**: {inputs.get('responsibilities', '')}
- **Preferred Qualifications**: {inputs.get('preferred_qualifications', '')}
- **Company Info**: {inputs.get('company_info', '')}

Ensure:
- No mention of gender, age, race, religion, or physical ability
- Use inclusive language (e.g., "you’ll work with..." instead of "he/she will...")
- Avoid corporate jargon or overly aggressive language
- Emphasize inclusivity, collaboration, growth mindset, and impact

Return the unbiased job description only. Do not include explanations.
"""

    def create_unbiased_jd(self, inputs: Dict[str, str]) -> str:
        prompt = self.generate_prompt(inputs)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            jd = response.choices[0].message.content.strip()
            return jd
        except Exception as e:
            print("Error generating JD:", e)
            return "Sorry, we couldn't generate the job description at this time."

    def detect_biased_language(self, company_info: str) -> Dict[str, List[str]]:
        flagged = {category: [] for category in self.bias_indicators}
        lower_info = company_info.lower()

        for category, keywords in self.bias_indicators.items():
            for word in keywords:
                if re.search(rf"\b{re.escape(word)}\b", lower_info):
                    flagged[category].append(word)

        return {k: v for k, v in flagged.items() if v}

    def ai_detect_biased_language(self, company_info: str) -> str:
        prompt = f"""
You are an expert in inclusive workplace language. Analyze the following company culture description for **any biased, exclusionary, or problematic language** related to gender, age, physical ability, or culture. 

- Return a **bullet list** of phrases or words that may be biased.
- Briefly explain **why each is problematic**.
- Focus on subtle language too, not just obvious slurs.

Text:
\"\"\"{company_info}\"\"\"
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error detecting bias using AI:", e)
            return "Could not analyze company culture at this time."
