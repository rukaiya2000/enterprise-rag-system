import os
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
PROCESSED_FOLDER = Path("../data/processed")
PROCESSED_FOLDER.mkdir(exist_ok=True)

EMBEDDING_DIM = 384  # MiniLM-L6 dimension

# -----------------------------
# Load chunks
# -----------------------------
all_texts = []
all_metadata = []

for file_path in PROCESSED_FOLDER.iterdir():
    if file_path.suffix.lower() != ".json":
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for chunk in chunks:
        all_texts.append(chunk["text"])
        all_metadata.append({
            "policy_id": chunk["policy_id"],
            "policy_name": chunk["policy_name"],
            "section": chunk["section"],
            "source_file": chunk["source_file"]
        })

print(f"Loaded {len(all_texts)} chunks from {len(list(PROCESSED_FOLDER.iterdir()))} files.")

# -----------------------------
# Generate embeddings
# -----------------------------
print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_texts, show_progress_bar=True)

# -----------------------------
# Create FAISS index
# -----------------------------
index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(np.array(embeddings, dtype='float32'))

print(f"FAISS index created with {len(all_texts)} vectors.")

# -----------------------------
# Save index + metadata
# -----------------------------
faiss.write_index(index, f'{PROCESSED_FOLDER / "policy_index.faiss"}')
np.save(PROCESSED_FOLDER / "metadata.npy", all_metadata)

print("FAISS index and metadata saved in 'data/processed/'")
