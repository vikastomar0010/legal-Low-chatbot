# 🏛️ AI-Powered Legal Law Chatbot (Minor Research Project – 2025)

FastAPI Backend • Streamlit Frontend • RAG • Legal Advice Retrieval • Location-Based Assistance

---

## 📌 Overview

This project is a full-stack AI Legal Chatbot designed to assist users with questions related to Indian law.

It combines:

- Retrieval-Augmented Generation (RAG)
- Real lawyer advice retrieval
- Location-based legal assistance

The system provides both legal context (statutes, punishments, applicability) and practical guidance (real lawyer advice + nearby legal resources).

---

## 🏗️ System Architecture

- FastAPI – Backend APIs
- Streamlit – Frontend Interface
- Fine-tuned INLegalBERT – Legal semantic embeddings
- FAISS – High-speed vector search
- BART Large – Legal Q&A summarization
- Mistral-7B (Together API) – Legal explanation generation

---

# 🚀 Key Functionalities

---

## 1️⃣ Legal Knowledge Retrieval (RAG)

This module provides structured law-based explanations using IPC, CrPC, HMA, and other statutes.

### How it Works:
- Retrieves relevant legal text using FAISS
- Uses Mistral-7B to generate a structured 5-point legal answer

### Capabilities:
- Identifies correct legal sections
- Explains punishment & exceptions
- Describes applicability
- Generates concise structured summaries

---

## 2️⃣ Legal Advice Retrieval (Real Lawyer Answers)

Retrieves real lawyer-provided answers by matching user queries with summarized legal Q&A data.

### Pipeline:
- Fine-tuned INLegalBERT embeddings
- FAISS for fast similarity search
- CrossEncoder for reranking precision

### Provides:
- Best matching lawyer advice
- Advice summary
- Source URL
- Similarity score
- Rerank score

---

## 3️⃣ Location-Based Legal Assistance

When a user provides a city name, the system can help locate:

- Lawyers
- Legal aid centers
- Police stations
- Courts

---

# 📂 Project Structure
legal_chat_bot/
│
- ├── backend/
- │ ├── main.py
- │ ├── rag_module.py
- │ ├── advice_module.py
- │ ├── models/
- │ │ └── inlegalbert_model.py
- │ ├── data/
- │ │ ├── processed_lawyer_data.json
- │ │ ├── summarized_legal_data.json
- │ │ ├── legal_chunks.npy
- │ │ ├── legal_embeddings.npy
- │ │ ├── question_embeddings.pt
- │ │ └── faiss_index/
- │ ├── scripts/
- │ │ ├── build_legal_chunks.py
- │ │ ├── build_question_embeddings.py
- │ │ ├── build_advice_summaries.py
- │ │ ├── fine_tune_inlegalbert.py
- │ │ └── build_advice_index.py
- │
- ├── frontend/
- │ └── app.py
- │
- ├── .gitignore
- └── README.md

## Tech Stack
- Python
- FastAPI
- Streamlit
- SentenceTransformers
- INLegalBERT
- BART Large
- FAISS
- CrossEncoder
- Together API
- Mistral-7B

---
📌 Future Roadmap

- Chat history support
- Improved nearby legal centers detection
- Docker deployment
- Hindi / multilingual support
- Local LLM integration
---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legal_chat_bot.git
cd legal_chat_bot





