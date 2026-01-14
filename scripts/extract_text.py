import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw_pdfs")
OUT_DIR = Path("data/extracted_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    out_path.write_text(text, encoding="utf-8")

for class_dir in RAW_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    for pdf in tqdm(class_dir.rglob("*.pdf"), desc=f"Processing {class_dir.name}"):
        rel_path = pdf.relative_to(RAW_DIR)
        out_file = OUT_DIR / rel_path.with_suffix(".txt")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        extract_pdf(pdf, out_file)
