# backend/rag/ingest.py
import os
from pathlib import Path
import json

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI()

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[2] / "index"
INDEX_DIR.mkdir(exist_ok=True)


# --- STEP 1: List both .txt and .pdf files ---
def list_input_files():
    return list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))


# --- STEP 2: Read text from either type ---
def read_file_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    elif file_path.suffix.lower() == ".pdf":
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")
        return text.strip()

    else:
        return ""


# --- STEP 3: Split into chunks ---
def chunk_text(text: str, max_chars: int = 400, overlap: int = 100):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]


# --- STEP 4: Embed text chunks (batched) ---
def embed_texts(texts, model: str = "text-embedding-3-small", batch_size: int = 8):
    vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} / {(total - 1) // batch_size + 1} (size={len(batch)})")
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype="float32")


# --- STEP 5: Build FAISS index ---
def build_index():
    files = list_input_files()
    if not files:
        print("No .txt or .pdf files found in docs/ — add some first.")
        return

    all_chunks = []
    metadata = []

    for f in files:
        print(f"Reading {f.name}...")
        text = read_file_text(f)
        if not text:
            print(f" No text extracted from {f.name}, skipping.")
            continue

        chunks = chunk_text(text, max_chars=400, overlap=100)
        print(f"  → {len(chunks)} chunks")
        for c in chunks:
            all_chunks.append(c)
            metadata.append({"source": f.name, "text": c})

    if not all_chunks:
        print("No chunks found after processing files. Nothing to index.")
        return

    print(f"Total chunks to embed: {len(all_chunks)}")

    # If you're still hitting memory issues, temporarily limit for testing:
    # all_chunks = all_chunks[:200]
    # metadata = metadata[:200]

    X = embed_texts(all_chunks, batch_size=8)
    dim = X.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(X)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("------ Index built in index/faiss.index and index/metadata.json -----")


if __name__ == "__main__":
    build_index()
