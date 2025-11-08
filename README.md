# TrustNote

TrustNote — Retrieval-Augmented Generation (RAG) API

TrustNote is a lightweight Retrieval-Augmented Generation (RAG) backend built with FastAPI, FAISS, and OpenAI’s API.
It lets you upload .txt or .pdf documents, embed them into a searchable vector index, and ask natural-language questions that are answered using your own document context.
Tech Stack

Python 3.10+
FastAPI
FAISS (semantic similarity search)
OpenAI API (embeddings + text generation)
dotenv (for environment configuration)

TrustNote/
├── backend/
│   ├── main.py          # FastAPI entry point
│   ├── rag/
│   │   ├── ingest.py    # Builds FAISS index from docs
│   │   └── engine.py    # Query engine (retrieval + generation)
├── docs/                # Place your .txt / .pdf files here
├── index/               # Stores faiss.index and metadata.json
└── README.md


Setup Instructions

git clone https://github.com/sandeepshabd/TrustNote.git
cd TrustNote
pip install -r requirements.txt
export OPENAI_API_KEY=your_openai_api_key_here

python -m backend.rag.ingest

This will:

Read your documents
Split them into text chunks
Generate embeddings
Build and save a FAISS index in the index/ folder

Embedding 120 chunks...
------ Index built in index/faiss.index and index/metadata.json -----


Start the FastAPI server:
uvicorn backend.main:app --reload --log-level debug

http://127.0.0.1:8000/docs
You’ll see an interactive Swagger UI where you can send questions like:
POST /ask
{
  "query": "What are the key insights from the Q3 report?"
}


Example Workflow

Add PDFs or text files to /docs.
Run python -m backend.rag.ingest to index them.
Start the app with Uvicorn.
Use /ask endpoint in Swagger UI to query your data.



