# backend/rag/engine.py
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --------- Logging setup (library-style) ---------
logger = logging.getLogger(__name__)
# DO NOT call basicConfig here in a library module; do it in your main app.
# Example in main: logging.basicConfig(level=logging.DEBUG)

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
        logger.debug("Initializing RAGEngine with k=%s", k)

        if not FAISS_PATH.exists() or not META_PATH.exists():
            logger.error(
                "Index or metadata not found. FAISS_PATH=%s exists=%s, META_PATH=%s exists=%s",
                FAISS_PATH,
                FAISS_PATH.exists(),
                META_PATH,
                META_PATH.exists(),
            )
            raise RuntimeError(
                "Index or metadata not found. "
                "Run `python -m backend.rag.ingest` first."
            )

        logger.debug("Loading FAISS index from %s", FAISS_PATH)
        self.index = faiss.read_index(str(FAISS_PATH))

        logger.debug("Loading metadata from %s", META_PATH)
        self.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

        logger.info(
            "RAGEngine initialized: %d metadata entries, k=%d",
            len(self.metadata),
            k,
        )
        self.k = k

    # ---------- Embedding & Retrieval ----------

    def _embed_query(self, query: str, model: str = "text-embedding-3-small") -> np.ndarray:
        logger.debug("Embedding query (model=%s): %s", model, query)

        try:
            resp = client.embeddings.create(
                model=model,
                input=[query],
            )
        except Exception as e:
            logger.exception("Error calling OpenAI embeddings API: %s", e)
            raise

        vec = np.array(resp.data[0].embedding, dtype="float32")
        logger.debug("Got embedding vector of shape %s", vec.shape)

        return np.expand_dims(vec, axis=0)

    def retrieve(self, query: str):
        """
        Return top-k docs (list of dicts: source, text, score).
        """
        logger.debug("Starting retrieval for query: %s", query)
        q_vec = self._embed_query(query)

        logger.debug("Searching FAISS index with k=%d", self.k)
        distances, indices = self.index.search(q_vec, self.k)

        logger.debug(
            "FAISS search done. distances=%s indices=%s",
            distances[0].tolist(),
            indices[0].tolist(),
        )

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta_idx = int(idx)
            if meta_idx < 0 or meta_idx >= len(self.metadata):
                logger.warning(
                    "FAISS returned invalid index %s (metadata length %d)",
                    meta_idx,
                    len(self.metadata),
                )
                continue

            meta = self.metadata[meta_idx]
            result = {
                "source": meta.get("source", "unknown"),
                "text": meta.get("text", ""),
                "score": float(dist),
            }
            results.append(result)

        logger.info("Retrieved %d docs for query", len(results))
        # Log a small preview of each doc
        for i, r in enumerate(results):
            preview = r["text"][:120].replace("\n", " ")
            logger.debug(
                "Doc %d | source=%s | score=%.4f | preview=%s...",
                i,
                r["source"],
                r["score"],
                preview,
            )

        return results

    # ---------- Prompt Building ----------

    def _build_messages(self, query: str, docs: list[dict]):
        logger.debug("Building messages for query and %d docs", len(docs))

        context = "\n\n---\n\n".join(d["text"] for d in docs)

        # To avoid massive logs, only log a snippet of the context
        context_preview = context[:500].replace("\n", " ")
        logger.debug("Context preview (first 500 chars): %s...", context_preview)

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

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        logger.debug("Messages built for Responses API call.")
        return messages

    # ---------- Responses API Call ----------

    def _call_responses(self, messages, model: str = "gpt-4o-mini") -> str:
        """
        Call OpenAI Responses API and extract plain text answer.
        """
        logger.info("Calling OpenAI Responses API with model=%s", model)

        # Optional: log just roles + first 100 chars of content
        for i, m in enumerate(messages):
            c_preview = str(m["content"])[:120].replace("\n", " ")
            logger.debug("Message %d | role=%s | content preview=%s...", i, m["role"], c_preview)

        try:
            resp = client.responses.create(
                model=model,
                input=messages,
            )
        except Exception as e:
            logger.exception("Error calling OpenAI Responses API: %s", e)
            raise

        logger.debug("Raw response object type: %s", type(resp))

        text_chunks = []

        # Newer SDKs have resp.output as a list of "message" items
        if hasattr(resp, "output") and resp.output:
            logger.debug("Parsing resp.output with %d items", len(resp.output))
            for item in resp.output:
                content_list = getattr(item, "content", []) or []
                for c in content_list:
                    # Some SDK versions use c.text, others c.output_text.text
                    if hasattr(c, "text") and c.text:
                        text_chunks.append(c.text)
                    elif hasattr(c, "output_text") and hasattr(c.output_text, "text"):
                        text_chunks.append(c.output_text.text)
        else:
            logger.warning("resp.output is missing or empty, trying fallback")

        # Fallback: try a common direct path
        if not text_chunks:
            logger.debug("Using fallback response parsing branch.")
            try:
                text_chunks.append(resp.output[0].content[0].text)
            except Exception as e:
                logger.exception("Fallback parsing failed; stringifying resp: %s", e)
                text_chunks.append(str(resp))

        answer = "".join(text_chunks).strip()
        logger.info("Got answer text of length %d", len(answer))
        logger.debug("Answer preview: %s...", answer[:300].replace("\n", " "))

        return answer

    # ---------- Public API ----------

    def answer(self, query: str, model: str = "gpt-4o-mini"):
        """
        High-level method:
        1) Retrieve top-k context chunks
        2) Build messages
        3) Call Responses API
        4) Return (answer_text, docs)
        """
        logger.info("RAGEngine.answer called with query: %s", query)

        docs = self.retrieve(query)
        messages = self._build_messages(query, docs)
        answer_text = self._call_responses(messages, model=model)

        logger.info("RAGEngine.answer completed. Returning answer and %d docs.", len(docs))
        return answer_text, docs
