import os
import json
from pathlib import Path
import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

RAW_FOLDER = Path("../data/raw")
PROCESSED_FOLDER = Path("../data/processed")
PROCESSED_FOLDER.mkdir(exist_ok=True)

CHUNK_SIZE = 500  # Approx tokens per chunk

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def ingest_pdf(file_path):
    reader = PdfReader(str(file_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def ingest_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n")

def clean_text(text):
    # Remove page numbers
    text = re.sub(r'Page \d+ of \d+', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text

def main():
    for file_path in RAW_FOLDER.iterdir():
        if file_path.suffix.lower() == ".pdf":
            text = ingest_pdf(file_path)
        elif file_path.suffix.lower() == ".html":
            text = ingest_html(file_path)
        else:
            continue

        text = clean_text(text)

        chunks = chunk_text(text)

        # Build metadata per chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "policy_id": f"{file_path.stem}_chunk{i+1}",
                "policy_name": file_path.stem,
                "section": f"section_{i+1}",
                "text": chunk,
                "version": "v1.0",
                "effective_date": "2026-01-01",
                "source_file": str(file_path)
            })

        # Save processed JSON
        out_file = PROCESSED_FOLDER / f"{file_path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(processed_chunks, f, indent=4)

        print(f"Processed {file_path.name}, {len(chunks)} chunks saved.")

if __name__ == "__main__":
    main()
