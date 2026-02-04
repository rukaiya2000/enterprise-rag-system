import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROCESSED_FOLDER = Path("../data/processed")
FAISS_INDEX_FILE = PROCESSED_FOLDER / "policy_index.faiss"
METADATA_FILE = PROCESSED_FOLDER / "metadata.npy"

# -----------------------------
# Load vector DB + metadata
# -----------------------------
index = faiss.read_index(str(FAISS_INDEX_FILE))
metadata = np.load(METADATA_FILE, allow_pickle=True).tolist()

# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# -----------------------------
# Helper functions
# -----------------------------
def retrieve_chunks(query, top_k=5):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype='float32'), top_k)

    results = []
    for idx in indices[0]:
        chunk_meta = metadata[idx]
        results.append({
            "policy_id": chunk_meta["policy_id"],
            "policy_name": chunk_meta["policy_name"],
            "section": chunk_meta["section"],
            "source_file": chunk_meta["source_file"]
        })
    return results


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    top_k = 5
    results = retrieve_chunks(user_query, top_k)

    print(f"\nTop {top_k} relevant chunks:\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['policy_name']} ({r['section']}) → {r['policy_id']} from {r['source_file']}")
