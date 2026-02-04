import json
from pathlib import Path

PROCESSED_FOLDER = Path("../data/processed")

REQUIRED_FIELDS = ["policy_id", "policy_name", "section", "text", "version", "effective_date", "source_file"]

for file_path in PROCESSED_FOLDER.iterdir():
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for chunk in data:
        for field in REQUIRED_FIELDS:
            assert field in chunk, f"{field} missing in {file_path.name}"
        assert len(chunk["text"]) > 50, f"Chunk too short in {file_path.name}"
print("============ All files validated successfully! ============")
