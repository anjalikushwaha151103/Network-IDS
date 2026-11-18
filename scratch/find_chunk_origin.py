"""Check where the numbers came from in earlier turns."""

import json
from pathlib import Path

path = Path(r"C:\Users\hp\.gemini\antigravity\brain\7381d497-bfee-4315-b215-58ba8313101f\.system_generated\logs\chunks\transcript\00000016.jsonl")
with open(path, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if any(x in line for x in ["85.34", "80.89", "67.24"]):
            d = json.loads(line)
            step_idx = d.get("step_index")
            src = d.get("source")
            tp = d.get("type")
            print(f"Line {idx} (step={step_idx}, src={src}, type={tp}):")
            for l in d.get("content", "").split("\n"):
                if any(x in l for x in ["85.34", "80.89", "67.24"]):
                    print("  ", l[:140])
