import re
from pathlib import Path
from tqdm import tqdm

TEXT_DIR = Path("data/extracted_text")

def clean_text(text):
    text = re.sub(r'\n\d+\n', '\n', text)  # page numbers
    text = re.sub(r'NCERT.*?\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

for txt_file in tqdm(list(TEXT_DIR.rglob("*.txt"))):
    raw = txt_file.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    txt_file.write_text(cleaned, encoding="utf-8")
