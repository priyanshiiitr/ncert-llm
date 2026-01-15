from pathlib import Path
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
VECTOR_DIR = Path("data/vector_store")
INDEX_FILE = VECTOR_DIR / "faiss.index"
META_FILE = VECTOR_DIR / "metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Lightweight, CPU-friendly model
LLM_NAME = "microsoft/phi-2"
TOP_K = 2
MAX_NEW_TOKENS = 256
# ==============================
def load_retriever():
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE,"r",encoding="utf-8") as f:
        metadata = json.load(f)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, embedder
def retrieve_context(query,index,metadata,embedder):
    qvec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(qvec, TOP_K)
    contexts = []
    for idx in indices[0]:
        meta = metadata[idx]
        contexts.append(
            f"(Class {meta['class']} {meta['subject']} – {meta['chapter']})"
        )
    return "\n".join(contexts), indices[0]
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def build_prompt(question, retrieved_context):
    return f"""
You are an NCERT textbook assistant.

STRICT RULES (VERY IMPORTANT):
- Answer ONLY using the NCERT context provided below.
- Do NOT use outside knowledge.
- Do NOT guess or add extra facts.
- Use simple, student-friendly language.
- If the answer is NOT clearly present in the NCERT context, say exactly:
  "This information is not found in the NCERT textbook."

NCERT CONTEXT:
{retrieved_context}

QUESTION:
{question}

ANSWER:
""".strip()

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,          # HARD LIMIT
            temperature=0.2,             # Less creativity
            do_sample=False,             # Deterministic
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Show only the answer
    if "ANSWER:" in text:
        return text.split("ANSWER:")[-1].strip()
    return text.strip()

def main():
    print("📘 NCERT Chatbot (type 'exit' to quit)\n")

    index, metadata, embedder = load_retriever()
    tokenizer, model = load_llm()

    while True:
        question = input("🧑‍🎓 Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        context_str, _ = retrieve_context(question, index, metadata, embedder)
        prompt = build_prompt(question, context_str)
        answer = generate_answer(prompt, tokenizer, model)

        print("\n🤖 Answer:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
