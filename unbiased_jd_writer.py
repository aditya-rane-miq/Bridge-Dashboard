import os
from typing import Dict
import openai

client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class UnbiasedJDWriter:
    def __init__(self):
        self.model = "mistralai/mistral-7b-instruct"

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
