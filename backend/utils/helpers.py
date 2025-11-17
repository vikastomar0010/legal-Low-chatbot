import torch

def save_embeddings(embeddings, questions, path="backend/data/question_embeddings.pt"):
    torch.save({"embeddings": embeddings, "questions": questions}, path)
    print(f"✅ Saved embeddings to {path}")

def load_embeddings(path="backend/data/question_embeddings.pt"):
    data = torch.load(path)
    return data["embeddings"], data["questions"]
