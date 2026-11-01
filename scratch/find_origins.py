import json

transcript_path = r"C:\Users\hp\.gemini\antigravity-ide\brain\8865a553-f8c6-474b-b88d-d1c8a8caa6b8\.system_generated\logs\transcript.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "46.54" in line or "82.26" in line or "74.34" in line or "2233" in line:
            obj = json.loads(line)
            source = obj.get("source")
            step = obj.get("step_index")
            content = obj.get("content", "")
            for term in ["46.54", "82.26", "74.34", "2233"]:
                if term in content:
                    idx = content.find(term)
                    snippet = content[max(0, idx-50):min(len(content), idx+50)].replace("\n", " ")
                    print(f"Step {step} ({source}) - [{term}]: ...{snippet}...")
