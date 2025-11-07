# backend/rag/engine.py
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.json"


class RAGEngine:
    """
    RAGEngine loads the FAISS index and metadata created by ingest.py
    and provides a method to answer questions using OpenAI Responses API.
    """

    def __init__(self, k: int = 4):
        if not FAISS_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(
                "Index or metadata not found. "
                "Run `python -m backend.rag.ingest` first."
            )

        self.index = faiss.read_index(str(FAISS_PATH))
        self.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.k = k

    # ---------- Embedding & Retrieval ----------

    def _embed_query(self, query: str, model: str = "text-embedding-3-small") -> np.ndarray:
        resp = client.embeddings.create(
            model=model,
            input=[query],
        )
        vec = np.array(resp.data[0].embedding, dtype="float32")
        return np.expand_dims(vec, axis=0)

    def retrieve(self, query: str):
        """
        Return top-k docs (list of dicts: source, text, score).
        """
        q_vec = self._embed_query(query)
        distances, indices = self.index.search(q_vec, self.k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[int(idx)]
            results.append(
                {
                    "source": meta.get("source", "unknown"),
                    "text": meta.get("text", ""),
                    "score": float(dist),
                }
            )
        return results

    # ---------- Prompt Building ----------

    def _build_messages(self, query: str, docs: list[dict]):
        context = "\n\n---\n\n".join(d["text"] for d in docs)

        system_msg = (
            "You are TrustNote, a helpful assistant. "
            "You answer questions strictly using the provided context. "
            "If the answer is not present in the context, say you don't know."
        )

        user_msg = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer clearly and concisely."
        )

        # Responses API expects a list of message dicts
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    # ---------- Responses API Call ----------

    def _call_responses(self, messages, model: str = "gpt-4o-mini") -> str:
        """
        Call OpenAI Responses API and extract plain text answer.
        """
        resp = client.responses.create(
            model=model,
            input=messages,
        )

        # Best-effort extraction of text from the response
        text_chunks = []

        # Newer SDKs have resp.output as a list of "message" items
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                content_list = getattr(item, "content", []) or []
                for c in content_list:
                    # Some SDK versions use c.text, others c.output_text.text
                    if hasattr(c, "text") and c.text:
                        text_chunks.append(c.text)
                    elif hasattr(c, "output_text") and hasattr(c.output_text, "text"):
                        text_chunks.append(c.output_text.text)

        # Fallback: try a common direct path
        if not text_chunks:
            try:
                text_chunks.append(resp.output[0].content[0].text)
            except Exception:
                # last resort: stringify
                text_chunks.append(str(resp))

        return "".join(text_chunks).strip()

    # ---------- Public API ----------

    def answer(self, query: str, model: str = "gpt-4o-mini"):
        """
        High-level method:
        1) Retrieve top-k context chunks
        2) Build messages
        3) Call Responses API
        4) Return (answer_text, docs)
        """
        docs = self.retrieve(query)
        messages = self._build_messages(query, docs)
        answer_text = self._call_responses(messages, model=model)
        return answer_text, docs
