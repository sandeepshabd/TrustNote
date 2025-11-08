# backend/main.py
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag.engine import RAGEngine  # engine.py you showed

# ---------- Logging setup ----------

# This is the "top-level" logging config for your app.
# All logger calls in backend.rag.engine will respect this.
logging.basicConfig(
    level=logging.DEBUG,  # change to INFO in prod
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- FastAPI app ----------

app = FastAPI(
    title="TrustNote RAG API",
    version="0.1.0",
    description="Simple RAG backend using FAISS + OpenAI Responses API",
)

# Instantiate RAGEngine at import time so we reuse it across requests
try:
    rag_engine = RAGEngine(k=4)
except RuntimeError as e:
    # This will show up in logs if index/metadata is missing
    logger.exception("Failed to initialize RAGEngine: %s", e)
    rag_engine = None


# ---------- Pydantic models ----------

class AskRequest(BaseModel):
    query: str
    model: str | None = "gpt-4o-mini"


class Doc(BaseModel):
    source: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    docs: list[Doc]


# ---------- Routes ----------

@app.get("/")
async def root():
    return {"status": "ok", "message": "TrustNote RAG API is running"}


@app.get("/health")
async def health():
    """
    Simple health check.
    Also tells you if RAGEngine initialized correctly.
    """
    if rag_engine is None:
        return {"status": "error", "detail": "RAGEngine failed to initialize"}
    return {"status": "ok", "detail": "RAGEngine ready"}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Main RAG endpoint.
    - Takes `query` and optional `model`
    - Returns answer text and the retrieved documents
    """
    if rag_engine is None:
        logger.error("RAGEngine is not initialized; cannot answer query.")
        raise HTTPException(
            status_code=500,
            detail="RAGEngine not initialized. Check index/metadata and server logs.",
        )

    logger.info("Received /ask request: %s", body.query)

    try:
        answer_text, docs = rag_engine.answer(
            query=body.query,
            model=body.model or "gpt-4o-mini",
        )
    except Exception as e:
        logger.exception("Error while generating answer: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Error while generating answer. See server logs for details.",
        )

    logger.info("Returning answer of length %d and %d docs", len(answer_text), len(docs))

    # docs is already a list[dict] with keys source/text/score, so it matches Doc
    return AskResponse(answer=answer_text, docs=docs)
