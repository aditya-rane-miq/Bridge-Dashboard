import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load model only once
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Get embedding vector
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def load_backend_data():
    path = os.path.join("Inclusive_Mentorship_Program_Raw_data.xlsx")
    # Debug: Print absolute path and check if file exists
    abs_path = os.path.abspath(path)
    df = pd.read_excel(abs_path)
    return df



def load_backend_mentor_data():
    path = os.path.join("Inclusive_Mentorship_Program_Raw_data.xlsx")
    abs_path = os.path.abspath(path)
    print("Looking for file at:", abs_path)
    print("File exists?", os.path.exists(abs_path))
    df = pd.read_excel(abs_path)

     # Normalize text-based fields
    df['Leadership_Potential'] = df['Leadership_Potential'].astype(str).str.strip().str.title()

    # Generate High_Potential_Flag based on defined logic
    df['High_Potential_Flag'] = np.where(
        (df['Performance_Rating'] >= 4) &
        (df['Leadership_Potential'].isin(['High', 'Medium'])),
        'Yes',
        'No'
    )

    df = df[df['High_Potential_Flag'] == 'Yes']

    return df
# Match logic
def match_employee_with_mentor(employee, mentors, tokenizer, model, mentorship_type="Mentorship"):
    matches = []
    emp_expert_embed = get_embeddings(employee["development_area"], tokenizer, model)
    emp_goal_embed = get_embeddings(employee["Career_Goal"], tokenizer, model)
    emp_combined = np.concatenate((emp_expert_embed, emp_goal_embed))
    emp_id_str = str(employee["EmpID"]).strip()  # Normalize for comparison

    emp_id_str = str(employee["EmpID"]).strip()
    emp_role = employee["Role_Level"]

    # ✅ Filter mentors: exclude employee's own record and match role level
    mentors_filtered = mentors[
        (mentors["Employee_ID"].astype(str).str.strip() != emp_id_str) &
        (mentors["Role_Level"].isin([emp_role]))
    ]
        # Run match logic
    for _, mentor in mentors_filtered.iterrows():
        mentor_expert_embed = get_embeddings(mentor["Preferred_Expertise"], tokenizer, model)
        mentor_goal_embed = get_embeddings(mentor["Career_Goal"], tokenizer, model)
        mentor_combined = np.concatenate((mentor_expert_embed, mentor_goal_embed))

        similarity = cosine_similarity([emp_combined], [mentor_combined])[0][0]
        matches.append({
            "Mentor ID": mentor.get("Mentor_Name", mentor.get("Employee_ID", "Unknown")),
            "Similarity Score": round(similarity, 2),
            "Department": mentor.get("Department", "N/A"),
            "Expertise": mentor.get("Preferred_Expertise", "N/A")
        })

         # 7️⃣ Sort by similarity and assign ranking
    sorted_matches = sorted(matches, key=lambda x: x["Similarity Score"], reverse=True)
    for idx, match in enumerate(sorted_matches, start=1):
        match["Ranking"] = idx
        del match["Similarity Score"]  # Remove similarity after ranking is added

    return sorted_matches


def match_employee_with_mentee(employee, mentors, tokenizer, model, mentorship_type="Reverse Mentorship"):
    matches = []
    emp_expert_embed = get_embeddings(employee["development_area"], tokenizer, model)
    emp_goal_embed = get_embeddings(employee["Career_Goal"], tokenizer, model)
    emp_combined = np.concatenate((emp_expert_embed, emp_goal_embed))


    emp_id_str = str(employee["EmpID"]).strip()
    emp_role = employee["Role_Level"]

    # ✅ Filter mentors: exclude employee's own record and match role level
    mentors_filtered = mentors[
        (mentors["Employee_ID"].astype(str).str.strip() != emp_id_str) &
        (mentors["Role_Level"].isin([emp_role]))
    ]
        # Run match logic
    for _, mentor in mentors_filtered.iterrows():
        mentor_expert_embed = get_embeddings(mentor["Preferred_Expertise"], tokenizer, model)
        mentor_goal_embed = get_embeddings(mentor["Career_Goal"], tokenizer, model)
        mentor_combined = np.concatenate((mentor_expert_embed, mentor_goal_embed))

        similarity = cosine_similarity([emp_combined], [mentor_combined])[0][0]
        matches.append({
            "Employee ID": mentor.get("Mentor_Name", mentor.get("Employee_ID", "Unknown")),
            "Similarity Score": round(similarity, 2),
            "Department": mentor.get("Department", "N/A"),
            "Expertise": mentor.get("Preferred_Expertise", "N/A")
        })
         # 7️⃣ Sort by similarity and assign ranking
    sorted_matches = sorted(matches, key=lambda x: x["Similarity Score"], reverse=True)
    for idx, match in enumerate(sorted_matches, start=1):
        match["Ranking"] = idx
        del match["Similarity Score"]  # Remove similarity after ranking is added

    return sorted_matches





def match_employee_with_training(employee, mentors, tokenizer, model):
    # 1️⃣ Combine dev area + career goal for the employee
    emp_text = f"{employee['development_area']}, {employee['Career_Goal']}"
    emp_embed = get_embeddings(emp_text, tokenizer, model)

    emp_id_str = str(employee["EmpID"]).strip()
    emp_role = employee["Role_Level"]

    # 2️⃣ Filter mentors (exclude self and match role level)
    mentors_filtered = mentors[
        (mentors["Employee_ID"].astype(str).str.strip() != emp_id_str) &
        (mentors["Role_Level"].isin([emp_role]))
    ]

    # 3️⃣ Drop rows with missing or empty Completed_Trainings
    mentors_filtered = mentors_filtered[mentors_filtered["Completed_Trainings"].notna()]
    mentors_filtered = mentors_filtered[mentors_filtered["Completed_Trainings"].str.strip() != ""]

    # 4️⃣ Compute embeddings and similarities
    matches = []
    for _, row in mentors_filtered.iterrows():
        training_text = row["Completed_Trainings"].strip()
        mentor_embed = get_embeddings(training_text, tokenizer, model)
        if mentor_embed is None or mentor_embed.shape != emp_embed.shape:
            continue

        similarity = cosine_similarity([emp_embed], [mentor_embed])[0][0]
        matches.append({
            "Assigned Training Courses": training_text,
            "Similarity Score": round(similarity, 2)
        })

    # 5️⃣ Remove duplicates based on training content and assign ranking
    sorted_matches = sorted(matches, key=lambda x: x["Similarity Score"], reverse=True)
    unique_trainings = {}
    for match in sorted_matches:
        training = match["Assigned Training Courses"]
        if training not in unique_trainings:
            unique_trainings[training] = match

    final_matches = list(unique_trainings.values())
    for idx, match in enumerate(final_matches, start=1):
        match["Ranking"] = idx
        del match["Similarity Score"]

    return final_matches
