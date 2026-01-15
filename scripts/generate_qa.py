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

LLM_NAME = "microsoft/phi-2"   # Use GPU in Colab
MAX_NEW_TOKENS = 200
MAX_PROMPT_TOKENS = 1800
MAX_CHUNK_TOKENS = 1200
# =========================================


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model


def truncate_by_tokens(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)


def build_prompt(chunk, meta):
    return f"""
You are generating training data for an NCERT-based AI tutor.

TASK:
From the NCERT text below, generate ONE good question-answer pair.

RULES:
- Question MUST be answerable from the given text.
- Answer MUST strictly use the given text.
- Use simple NCERT-style language.
- If the text is clearly unsuitable, respond with exactly:
  SKIP

METADATA:
Class: {meta['class']}
Subject: {meta['subject']}
Chapter: {meta['chapter']}

NCERT TEXT:
{chunk}

FORMAT (VERY IMPORTANT):
Q: <question>
A: <answer>
""".strip()


def generate(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    tokenizer, model = load_llm()

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    written = 0

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for record in tqdm(records, desc="Generating Q&A"):

            # 1️⃣ Truncate raw NCERT chunk
            safe_chunk = truncate_by_tokens(
                record["text"], tokenizer, MAX_CHUNK_TOKENS
            )

            # 2️⃣ Build prompt
            prompt = build_prompt(safe_chunk, record)

            # 3️⃣ Truncate FULL prompt
            prompt = truncate_by_tokens(
                prompt, tokenizer, MAX_PROMPT_TOKENS
            )

            # 4️⃣ Generate
            output = generate(prompt, tokenizer, model)

            if "SKIP" in output:
                continue

            try:
                question = output.split("Q:")[1].split("A:")[0].strip()
                answer = output.split("A:")[1].strip()
            except Exception:
                continue

            example = {
                "instruction": f"{question} (According to NCERT Class {record['class']} {record['subject']})",
                "input": "",
                "output": answer
            }

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print("✅ Dataset generation complete")
    print(f"📘 Total Q&A pairs created: {written}")


if __name__ == "__main__":
    main()
