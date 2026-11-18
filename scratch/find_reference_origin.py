"""Find the origin of the numbers quoted in the LR report."""

import json
from pathlib import Path

print("Checking JSON files in results...")
for p in Path("results").rglob("*.json"):
    try:
        data = json.load(open(p, encoding="utf-8"))
        if isinstance(data, dict):
            acc = data.get("accuracy")
            mf1 = data.get("macro_f1")
            wf1 = data.get("weighted_f1")
            if acc is not None:
                print(f"{p}: acc={acc}, macro_f1={mf1}, weighted_f1={wf1}")
    except Exception:
        pass

print("\nChecking markdown files in results...")
for p in Path("results").rglob("*.md"):
    try:
        text = p.read_text(encoding="utf-8")
        if "85.34" in text or "80.89" in text or "67.24" in text:
            print(f"Found match in {p}")
    except Exception:
        pass
