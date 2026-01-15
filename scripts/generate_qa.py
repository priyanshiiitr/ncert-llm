"""
Generate NCERT-style Q&A / instruction dataset from chunks.jsonl
"""

from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ================= CONFIG =================
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
OUT_DIR = Path("data/qa_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "ncert_qa.jsonl"

LLM_NAME = "microsoft/phi-2"   # Use GPU (Colab)
MAX_NEW_TOKENS = 200
# =========================================


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def truncate_text(text, tokenizer, max_tokens=1500):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def build_prompt(chunk, meta):
    return f"""
You are generating training data for an NCERT-based AI tutor.

TASK:
From the NCERT text below, generate EXACTLY:
1. One clear definition-type question
2. One clear explanation-type question

RULES:
- Questions must be answerable ONLY from the given text.
- Answers must be strictly based on the text.
- Use simple NCERT-style language.
- If the text is not suitable for questions, respond with ONLY:
  "SKIP"

METADATA:
Class: {meta['class']}
Subject: {meta['subject']}
Chapter: {meta['chapter']}

NCERT TEXT:
{chunk}

FORMAT:
Q1: <question>
A1: <answer>
Q2: <question>
A2: <answer>
""".strip()


def generate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    tokenizer, model = load_llm()

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    written = 0

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for record in tqdm(chunks, desc="Generating Q&A"):
            safe_chunk = truncate_text(record["text"], tokenizer)
            prompt = build_prompt(safe_chunk, record)

            output = generate(prompt, tokenizer, model)

            if "SKIP" in output:
                continue

            try:
                q1 = output.split("Q1:")[1].split("A1:")[0].strip()
                a1 = output.split("A1:")[1].split("Q2:")[0].strip()
                q2 = output.split("Q2:")[1].split("A2:")[0].strip()
                a2 = output.split("A2:")[1].strip()
            except Exception:
                continue

            examples = [
                {
                    "instruction": f"{q1} (According to NCERT Class {record['class']} {record['subject']})",
                    "input": "",
                    "output": a1
                },
                {
                    "instruction": f"{q2} (According to NCERT Class {record['class']} {record['subject']})",
                    "input": "",
                    "output": a2
                }
            ]

            for ex in examples:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1

    print(f"✅ Dataset generation complete")
    print(f"📘 Total Q&A pairs created: {written}")


if __name__ == "__main__":
    main()
