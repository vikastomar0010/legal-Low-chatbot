# backend/rag_module.py

import os
import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain_community.embeddings import HuggingFaceEmbeddings
import together
from backend.models.inlegalbert_model import QAModel, LegalEmbeddingModel
from dotenv import load_dotenv

# Load environment variables (for Together API key)
load_dotenv()
TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
together.api_key = TOGETHER_API_KEY

# =====================================================================
# ✅ 1. Load Models and Data Once (at backend startup)
# =====================================================================

# Paths (use relative to backend/)


QUESTION_EMB_PATH = "backend/data/question_embeddings.pt"
FAISS_INDEX_PATH = "backend/data/faiss_index/legal_faiss.index"
CHUNKS_PATH = "backend/data/legal_chunks.npy"
LEGAL_DB_PATH = "backend/data/legal_data.json"

# Load models
print("🔹 Loading models and data...")
model = QAModel()
embedding_model = LegalEmbeddingModel()


# Load data
data = torch.load(QUESTION_EMB_PATH, weights_only=False)
question_embeddings = data["embeddings"]
questions = data["questions"]

# Load FAISS index and chunks
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
legal_chunks = np.load(CHUNKS_PATH, allow_pickle=True)

# Load question-answer database
if os.path.exists(LEGAL_DB_PATH):
    with open(LEGAL_DB_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        answers = {item["question"]: item["answer"] for item in qa_data}
else:
    answers = {}

print("✅ RAG Module initialized successfully.")

# =====================================================================
# ✅ 2. Helper Functions
# =====================================================================

def find_closest_match(user_question: str):
    """Find closest matching QA question using SentenceTransformer."""
    user_embedding = model.encode(user_question)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)
    best_match_idx = similarities.argmax().item()

    best_match_question = questions[best_match_idx]
    similarity_score = similarities[0][best_match_idx].item()

    return best_match_question, similarity_score


def search_legal_docs(query: str, top_k: int = 5):
    """Search top-k relevant law chunks using FAISS + cosine re-ranking."""
    # Step 1: Find closest chunks using FAISS
    query_embedding = np.array(embedding_model.embed(query)).astype("float32").reshape(1, -1)

# Safety check
    if query_embedding.shape[1] != faiss_index.d:
        raise ValueError(
            f"❌ Dimension mismatch: Query dim = {query_embedding.shape[1]}, "
            f"FAISS index dim = {faiss_index.d}"
        )

    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_chunks = [legal_chunks[idx] for idx in indices[0]]

    # Step 2: Re-rank retrieved chunks with cosine similarity
    query_vector = torch.tensor(query_embedding, dtype=torch.float32)
    chunk_vectors = torch.tensor(
    np.array([
        embedding_model.embed([chunk])[0]
        for chunk in retrieved_chunks
    ]),
    dtype=torch.float32
    )


    scores = util.pytorch_cos_sim(query_vector, chunk_vectors)[0]
    ranked_chunks = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    best_chunk, _ = ranked_chunks[0]

    # Step 3: Generate answer using Together AI (Mistral-7B)
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    prompt = f"""
    As a legal chatbot specializing in Indian Penal Code, respond with accurate and structured legal information.

    Guidelines:
    - Respond in 5 bullet points covering distinct legal aspects.
    - First mention which law or section the context falls under.
    - Each point should reflect real legal provisions and clarify applicability.
    - Avoid unnecessary assumptions; stay factual.
    

    CONTEXT: {best_chunk}

    QUESTION: {query}
    ANSWER:
    """

    try:
        response = together.Complete.create(
            model=model_name,
            prompt=prompt,
            max_tokens=300,
            temperature=1
        )

        if "choices" in response and response["choices"]:
            return response["choices"][0]["text"].strip()
        else:
            return "⚠️ No response received from Together AI."

    except Exception as e:
        return f"❌ Error generating answer: {e}"


def get_answer(user_question: str):
    """Main function to handle QA retrieval pipeline."""
    best_match_question, similarity_score = find_closest_match(user_question)

    if similarity_score >= 0.7:
        # High similarity — use stored Q&A
        answer_text = answers.get(best_match_question, "Answer not found in database.")
        return {
            "type": "Q&A Match",
            "question": best_match_question,
            "similarity": round(similarity_score, 2),
            "answer": answer_text
        }

    elif 0.3 < similarity_score < 0.7:
        # Moderate similarity — use document retrieval
        generated_answer = search_legal_docs(user_question)
        return {
            "type": "RAG Generation",
            "similarity": round(similarity_score, 2),
            "generated_answer": generated_answer
        }

    else:
        # No good match
        return {
            "type": "No Match",
            "similarity": round(similarity_score, 2),
            "message": "Sorry, no relevant legal information found at the moment."
        }
