from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from typing import Optional, List
import asyncio
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure working directory
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# LLM model function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMBEDDING_MODEL,
    )

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim

# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=embedding_func,
    ),
)

# Data models
class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None

# API routes
@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        result = await rag.aquery(
            request.query,
            param=QueryParam(
                mode=request.mode,
                only_need_context=request.only_need_context
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents", response_model=Response)
async def add_document(document: DocumentRequest):
    """Add a single document to the RAG system"""
    try:
        await rag.ainsert(document.content)
        return Response(
            status="success",
            message="Document inserted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/batch", response_model=Response)
async def add_documents(documents: List[DocumentRequest]):
    """Add multiple documents to the RAG system"""
    try:
        contents = [doc.content for doc in documents]
        await rag.ainsert(contents)
        return Response(
            status="success",
            message=f"Successfully inserted {len(documents)} documents"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/file", response_model=Response)
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(default="text")
):
    """Upload a file to the RAG system"""
    try:
        content = await file.read()

        # Handle different file types
        if file_type == "text":
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                text_content = content.decode('latin-1')
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_type}"
            )

        await rag.ainsert(text_content)

        return Response(
            status="success",
            message=f"Successfully processed file: {file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
