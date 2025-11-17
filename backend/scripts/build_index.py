import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.faiss_helper import create_faiss_index

embeddings = np.load("backend/data/legal_embeddings.npy", allow_pickle=True)
create_faiss_index(embeddings)
