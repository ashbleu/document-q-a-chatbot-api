# FastAPI application

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from get_context import getContext
from qa_model import Model
from preprocess import clearDocs
from typing import List

app = FastAPI()

class ChatRequest(BaseModel):
    document_urls: List[str]
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        context = getContext(request.document_urls, request.question)
        model = Model()
        response = model.ask(request.question, context)
        clearDocs()
        return {
            "document_urls": request.document_urls,
            "question": request.question,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
