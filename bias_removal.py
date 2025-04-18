import os
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from io import BytesIO
from typing import List
from dotenv import load_dotenv
import openai
import json

load_dotenv()

client = openai.OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)


class BlindRecruitmentProcessor:
    def extract_text_from_pdf(self, uploaded_file) -> str:
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def detect_bias_terms_with_llm(self, text: str) -> List[str]:
        prompt = (
            "You are an AI fairness auditor reviewing resumes. From the resume text below, extract a Python list of all words or phrases "
        "that could introduce bias in the hiring process. Don't include any skills like EQL, excel in your list. This includes, but is not limited to:\n"
        "- Full names or partial names\n"
        "- Gender-related terms (he, she, Mr., Ms., etc.)\n"
        "- Email addresses (especially those ending in @gmail.com or any domain)\n"
        "- Phone numbers\n"
        "- Nationalities or ethnicities (e.g., Indian, Hispanic, Asian)\n"
        "- Religious references (e.g., Christian, Muslim, Sikh, church)\n"
        "- Family or marital status (e.g., married, single, mother, father)\n"
        "- Locations (cities, states, countries)\n"
        "- Languages spoken (e.g., Hindi, Spanish)\n"
        "Resume:\n"
        f"{text[:1500]}\n\n"
        "Return only a Python list of the extracted words or phrases that match any of the above categories. Do not explain anything else.")
        
        try:
            response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            print("LLM Response:", content)
            result = eval(content) if content.startswith("[") else []
            return result if isinstance(result, list) else []
        except Exception as e:
            print("LLM Error:", e)
            return []

    def redact_pdf(self, uploaded_file, bias_terms: List[str]) -> BytesIO:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            for term in bias_terms:
                instances = page.search_for(term)
                for inst in instances:
                    page.add_redact_annot(inst, fill=(0, 0, 0))
            page.apply_redactions()

        redacted_io = BytesIO()
        doc.save(redacted_io)
        redacted_io.seek(0)
        return redacted_io

    def summarize_resume(self, text: str) -> dict:
        prompt = (
            "You are a resume summarizer AI. Don't include anything the could reveal the identity, gender or ethnicity. Based on the following resume text, extract the key information in JSON format with these fields:\n\n"
            "- Summary: Short paragraph summarizing the candidate\n"
            "- Suitable Roles: List of appropriate job titles\n"
            "- Education: List of degrees and institutions. Use this format - [Degree Name 1 -Institution Name 1]\n"
            "- Experience: List of job titles and companies. Use this format - [Experience Name 1 -Company Name 1]\n"
            "- Projects: List of major projects\n"
            "- Skills: List of skills\n"
            "- Certifications: If any\n"
            f"Resume:\n{text[:2000]}\n\n"
            "Return only valid JSON, no explanation. and if any field is empty dont even provide any empty list, dont provide any output for that field."
        )
        try:
            response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            content = response.choices[0].message.content.strip()
            print("LLM Response:", content)
            result = json.loads(content)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"Error during resume summarization: {e}")
            return {}


    def process_resume(self, uploaded_file) -> tuple:
        uploaded_file.seek(0)
        text = self.extract_text_from_pdf(uploaded_file)
        bias_terms = self.detect_bias_terms_with_llm(text)
        summary_info = self.summarize_resume(text)
        uploaded_file.seek(0)
        redacted_pdf = self.redact_pdf(uploaded_file, bias_terms)
        return text, redacted_pdf, bias_terms, summary_info
