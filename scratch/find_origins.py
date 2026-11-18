"""Search transcript for origin of 85.34% and 67.24%."""

import json

transcript_path = r"C:/Users/hp/.gemini/antigravity/brain/7381d497-bfee-4315-b215-58ba8313101f/.system_generated/logs/transcript.jsonl"

with open(transcript_path, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        d = json.loads(line)
        step = d.get("step_index")
        if step in range(1275, 1283):
            tp = d.get("type")
            print(f"=== Step {step} ({tp}) ===")
            txt = d.get("content", "")
            for l in txt.split("\n"):
                if any(k in l for k in ["Accuracy", "LGBM", "LightGBM", "Macro F1", "Weighted F1"]):
                    print("  ", l[:120])
