import faiss
import numpy as np
import os

def create_faiss_index(embeddings, path="backend/data/faiss_index/legal_faiss.index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print("✅ FAISS index saved:", path)

def load_faiss_index(path="backend/data/faiss_index/legal_faiss.index"):
    return faiss.read_index(path)

def search_faiss(index, query_vector, top_k=3):
    distances, indices = index.search(np.array([query_vector]).astype("float32"), top_k)
    return distances, indices
