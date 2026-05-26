"""
Split NIAH 1m.txt into 32k/64k/128k/256k/512k variants.
- Strip all newlines from the haystack
- Truncate haystack proportionally to target token size
- Append the instruction line at the end (separated by a single space)

Usage:
    python3 split_niah.py
"""
import os

SRC = "/home/jiangjiandong.jjd/tokenspeed/prompts/NIAH/1m.txt"
OUT_DIR = "/home/jiangjiandong.jjd/tokenspeed/prompts/NIAH"

INSTRUCTION_MARK = "You are a helpful AI bot"
TARGET_SIZES_K = [32, 64, 128, 256, 512, 1024]   # 1024k == 1m

# NIAH needle: a sentence directly answering the question, inserted at NEEDLE_DEPTH
NEEDLE = (
    " The best thing to do in San Francisco is to eat a sandwich "
    "and sit in Dolores Park on a sunny day. "
)
NEEDLE_DEPTH = 0.5   # 0.0 = front, 0.5 = middle, 1.0 = end

with open(SRC, "r", encoding="utf-8") as f:
    full_text = f.read()

# Split haystack vs instruction (instruction starts at the marker line)
idx = full_text.find(INSTRUCTION_MARK)
if idx < 0:
    raise RuntimeError(f"Cannot locate instruction marker '{INSTRUCTION_MARK}'")

haystack = full_text[:idx].rstrip()
instruction = full_text[idx:].strip()

# Strip ALL newlines from haystack -> single line; collapse 2+ spaces
haystack_one_line = haystack.replace("\r", " ").replace("\n", " ")
while "  " in haystack_one_line:
    haystack_one_line = haystack_one_line.replace("  ", " ")
haystack_one_line = haystack_one_line.strip()

# Strip newlines from instruction too (single line)
instruction_one_line = " ".join(instruction.split())

baseline_chars = len(haystack_one_line)
print(f"[info] haystack chars (after stripping newlines) = {baseline_chars}")
print(f"[info] instruction chars = {len(instruction_one_line)}")

# 1m baseline = full haystack
BASELINE_K = 1024

for size_k in TARGET_SIZES_K:
    if size_k == BASELINE_K:
        truncated = haystack_one_line
        out_name = "1m.txt"
    else:
        ratio = size_k / BASELINE_K
        n_chars = int(baseline_chars * ratio)
        truncated = haystack_one_line[:n_chars]
        out_name = f"{size_k}k.txt"

    # Insert needle at NEEDLE_DEPTH (split on a space boundary to avoid breaking words)
    insert_pos = int(len(truncated) * NEEDLE_DEPTH)
    space_pos = truncated.find(" ", insert_pos)
    if space_pos < 0:
        space_pos = insert_pos
    truncated_with_needle = truncated[:space_pos] + NEEDLE + truncated[space_pos:]

    final = truncated_with_needle + " " + instruction_one_line
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final)
    nl_count = final.count("\n")
    needle_hit = final.count(NEEDLE.strip())
    print(f"  -> {out_path}  chars={len(final)}  newlines={nl_count}  needle_count={needle_hit}")

print("[done]")
