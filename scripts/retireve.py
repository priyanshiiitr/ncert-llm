"""
Retrieve relevant NCERT chunks for a query.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ============== CONFIG =================
VECTOR_DIR = Path("data/vector_store")
INDEX_FILE = VECTOR_DIR / "faiss.index"
META_FILE = VECTOR_DIR / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
# ======================================


def main():
    query = input("🔎 Enter your question: ")

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([query]).astype("float32")

    index = faiss.read_index(str(INDEX_FILE))

    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    distances, indices = index.search(query_vec, TOP_K)

    print("\n📚 Retrieved NCERT Context:\n")

    for rank, idx in enumerate(indices[0]):
        meta = metadata[idx]
        print(f"--- Result {rank+1} ---")
        print(f"Class: {meta['class']} | Subject: {meta['subject']} | Chapter: {meta['chapter']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
