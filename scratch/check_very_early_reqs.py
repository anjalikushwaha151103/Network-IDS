import json

transcript_path = r"C:\Users\hp\.gemini\antigravity-ide\brain\8865a553-f8c6-474b-b88d-d1c8a8caa6b8\.system_generated\logs\transcript.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 200 and "OpenMax" in line and '"type":"USER_INPUT"' in line:
            obj = json.loads(line)
            print(f"=== User Input at line {i} ===")
            print(obj.get("content"))
