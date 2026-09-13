"""
Run this whenever you add/change files in data/source_docs/.

    python scripts/ingest.py
"""
from app.ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    count = run_ingestion()
    print(f"Indexed {count} chunks.")
