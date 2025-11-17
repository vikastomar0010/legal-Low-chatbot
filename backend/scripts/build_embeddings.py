import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.inlegalbert_model import LegalEmbeddingModel

chunks = np.load("backend/data/legal_chunks.npy", allow_pickle=True)
embedder = LegalEmbeddingModel()
embeddings = embedder.embed(chunks)
np.save("backend/data/legal_embeddings.npy", embeddings)
print("✅ Embeddings saved.")
