# 🏛️ AI-Powered Legal Law Chatbot (Minor Research Project – 2025)

An AI-powered **Legal Law Chatbot** designed to answer queries related to legal concepts, case laws, and statutory information.  
This project uses **semantic search**, **vector embeddings**, and **FAISS** to deliver accurate and context-aware responses.  
Built as a modular research prototype with future integration planned for **Streamlit/Gradio**.

---

## 🚀 Project Features

### 🔍 1. Semantic Search on Legal Documents
- Uses **Sentence-Transformers** to convert legal texts into dense embeddings.
- Retrieves the most relevant legal case laws based on query similarity.

### ⚡ 2. FAISS-Based Vector Database
- FAISS (Facebook AI Similarity Search) used for scalable similarity search.
- Optimized for large-scale retrieval of Indian legal case laws & statutes.

### 🤖 3. Transformer-Based Natural Language Understanding
- Uses pre-trained Transformer models for better contextual understanding.
- Improves accuracy and relevance of chatbot responses.

### 🧩 4. Modular and Scalable Pipeline
- Data preprocessing module  
- Embedding generation module  
- FAISS index builder  
- Semantic search + ranking module  
- Chatbot response pipeline  

### 🌐 5. Future Enhancements
- Deployment using Streamlit or Gradio  
- Integration of RAG (Retrieval Augmented Generation)  
- Multi-turn conversational memory  
- Larger database of case laws  

---

## 📁 Project Structure

├── data/
│   ├── raw/                 # Raw legal text/case laws
│   ├── processed/           # Cleaned & preprocessed text
│
├── embeddings/
│   └── faiss_index.bin      # Trained FAISS index
│
├── src/
│   ├── preprocessing.py     # Text cleaning, normalization
│   ├── embedder.py          # Embedding generation using Sentence-Transformers
│   ├── build_faiss.py       # Building FAISS index
│   ├── semantic_search.py   # Retrieval logic
│   ├── chatbot.py           # Core chatbot pipeline
│
├── app/
│   └── demo.ipynb           # Notebook demonstration
│
├── README.md
└── requirements.txt


---

## 🛠️ Tech Stack

| Component | Technology Used |
|----------|-----------------|
| Embeddings | Sentence-Transformers |
| Vector Store | FAISS |
| NLP Models | HuggingFace Transformers |
| Backend | Python |
| Deployment (future) | Streamlit / Gradio |
| Storage | Filesystem / optional cloud |

---

## 🔧 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/legal-law-chatbot.git
cd legal-law-chatbot
