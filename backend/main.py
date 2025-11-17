from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_module import get_answer


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    result = get_answer(request.query)
    return {"query": request.query, "result": result}
