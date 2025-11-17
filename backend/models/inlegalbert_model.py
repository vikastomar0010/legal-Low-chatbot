from sentence_transformers import SentenceTransformer
import numpy as np

class QAModel:
    """Handles question embeddings"""
    def __init__(self, model_name="sentence-transformers/msmarco-distilbert-base-v4"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """Encodes text(s) for question similarity search"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True)

class LegalEmbeddingModel:
    """Handles law text embeddings"""
    def __init__(self, model_name="law-ai/InLegalBERT"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """Encodes legal text(s) for FAISS search"""
        if isinstance(texts, str):
            texts = [texts]
        # Each vector = (768,)
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")
