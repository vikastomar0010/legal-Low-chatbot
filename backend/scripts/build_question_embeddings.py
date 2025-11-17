# backend/scripts/build_question_embeddings.py

import sys, os, json, numpy as np, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.inlegalbert_model import QAModel
from utils.helpers import save_embeddings

# ---------------------------------------------------------------------
# ✅ Paths
# ---------------------------------------------------------------------
DATA_PATH = "backend/data/legal_data.json"
OUTPUT_PATH = "backend/data/question_embeddings.pt"

# ---------------------------------------------------------------------
# ✅ Load QA dataset
# ---------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ QA data not found at {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
answers = {item["question"]: item["answer"] for item in qa_data}

print(f"🔹 Loaded {len(questions)} questions from {DATA_PATH}")

# ---------------------------------------------------------------------
# ✅ Initialize Q&A embedding model
# ---------------------------------------------------------------------
qa_model = QAModel()

# ---------------------------------------------------------------------
# ✅ Generate embeddings
# ---------------------------------------------------------------------
print("🔹 Generating question embeddings...")
question_embeddings = qa_model.encode(questions)



# ---------------------------------------------------------------------
# ✅ Save embeddings
# ---------------------------------------------------------------------
save_embeddings(question_embeddings, questions, path=OUTPUT_PATH)
print("✅ Question embeddings created and saved successfully.")
