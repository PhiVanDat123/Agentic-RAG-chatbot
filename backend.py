from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
from io import BytesIO
import PyPDF2
import docx

from rag_agent import AdvancedRAGAgent
from utils import extract_text_from_file

app = FastAPI(title="Agentic RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

# Global RAG agent instance
rag_agent = None

@app.on_event("startup")
async def startup_event():
    global rag_agent
    rag_agent = AdvancedRAGAgent()
    await rag_agent.initialize()
    print("RAG Agent initialized and ready!")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for the knowledge base"""
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        
        # Add to vector store
        await rag_agent.add_document(text, file.filename)
        
        return {"message": f"Document {file.filename} added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for the RAG agent"""
    try:
        response = await rag_agent.query(request.query)
        return ChatResponse(response=response["answer"], sources=response["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent_initialized": rag_agent is not None}

async def start_server():
    """Start the FastAPI backend server"""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_server())