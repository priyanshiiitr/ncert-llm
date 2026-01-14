"""
Build FAISS vector database from NCERT chunks.
"""

from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============== CONFIG =================
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
VECTOR_DIR = Path("data/vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTOR_DIR / "faiss.index"
META_FILE = VECTOR_DIR / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ======================================


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.jsonl not found. Run chunk_text.py first.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    metadata = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metadata.append({
                "class": record["class"],
                "subject": record["subject"],
                "chapter": record["chapter"],
                "chunk_id": record["chunk_id"]
            })

    print("🔢 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Vector DB built successfully!")
    print(f"📦 Total chunks indexed: {len(texts)}")


if __name__ == "__main__":
    main()
