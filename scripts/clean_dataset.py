from pathlib import Path
import json
import re

# ================= CONFIG =================
INPUT_FILE = Path("data/qa_dataset/ncert_qa.jsonl")
OUTPUT_FILE = Path("data/qa_dataset/ncert_qa_clean.jsonl")

MAX_ANSWER_CHARS = 600        # prevent hallucinated long answers
MIN_ANSWER_CHARS = 10         # avoid empty/weak answers

VAGUE_PATTERNS = [
    r"in the text",
    r"in the given figure",
    r"based on the figure",
    r"shown above",
]

# =========================================


def is_vague_question(q):
    q_lower = q.lower()
    return any(re.search(p, q_lower) for p in VAGUE_PATTERNS)


def clean_answer(ans):
    # Remove extra generated questions
    if "\nQ:" in ans:
        ans = ans.split("\nQ:")[0]

    # Remove trailing junk
    ans = ans.strip()

    return ans


def main():
    kept = 0
    dropped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            try:
                ex = json.loads(line)
            except Exception:
                dropped += 1
                continue

            q = ex["instruction"]
            a = ex["output"]

            # 🔴 Clean answer
            a = clean_answer(a)

            # 🔴 Filters
            if is_vague_question(q):
                dropped += 1
                continue

            if len(a) < MIN_ANSWER_CHARS:
                dropped += 1
                continue

            if len(a) > MAX_ANSWER_CHARS:
                dropped += 1
                continue

            if "Q:" in a:
                dropped += 1
                continue

            # ✅ Keep example
            ex["output"] = a
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print("✅ Dataset cleaning complete")
    print(f"📘 Kept examples   : {kept}")
    print(f"🗑️ Dropped examples: {dropped}")


if __name__ == "__main__":
    main()
