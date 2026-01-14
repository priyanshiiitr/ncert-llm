"""
Chunk NCERT chapter-wise text files into LLM-friendly chunks
with rich metadata (class, subject, chapter, chunk_id).

Input:
    data/extracted_text/class_x/{science,maths}/chapter.txt

Output:
    data/chunks/chunks.jsonl
"""

from pathlib import Path
import json
import re
from tqdm import tqdm

# ================= CONFIG =================
TEXT_DIR = Path("data/extracted_text")
OUT_DIR = Path("data/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "chunks.jsonl"

# Chunk size control (approx words, safer than tokens for now)
MIN_WORDS = 150
MAX_WORDS = 350
# ==========================================


def clean_paragraphs(text: str):
    """Split text into clean paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    return paragraphs


def chunk_paragraphs(paragraphs):
    """Group paragraphs into size-bounded chunks."""
    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        words = len(para.split())

        if current_words + words > MAX_WORDS:
            if current_words >= MIN_WORDS:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0

        current_chunk.append(para)
        current_words += words

    if current_chunk and current_words >= MIN_WORDS:
        chunks.append(" ".join(current_chunk))

    return chunks


def infer_metadata(txt_path: Path):
    """
    Extract class, subject, chapter name from path.
    Example:
    data/extracted_text/class_6/science/chapter7.txt
    """
    parts = txt_path.parts
    class_name = parts[2].replace("class_", "")
    subject = parts[3]
    chapter = txt_path.stem.replace("_", " ").title()

    return class_name, subject, chapter


def main():
    all_txt_files = list(TEXT_DIR.rglob("*.txt"))

    if not all_txt_files:
        print("❌ No extracted text files found. Run extract_text.py first.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for txt_file in tqdm(all_txt_files, desc="Chunking NCERT text"):
            raw_text = txt_file.read_text(encoding="utf-8")

            paragraphs = clean_paragraphs(raw_text)
            chunks = chunk_paragraphs(paragraphs)

            class_name, subject, chapter = infer_metadata(txt_file)

            for idx, chunk in enumerate(chunks):
                record = {
                    "class": class_name,
                    "subject": subject,
                    "chapter": chapter,
                    "chunk_id": idx,
                    "text": chunk,
                    "source": "NCERT"
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Chunking complete. Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
