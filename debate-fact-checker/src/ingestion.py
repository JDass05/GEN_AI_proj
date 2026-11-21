"""
Data ingestion: download SciFact, parse JSONL, chunk documents, build dual indices.

Usage:
    python -m src.ingestion
"""

import json
import os
import pickle
import random
import re
import tarfile
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# 1. Download & Extract SciFact
# ---------------------------------------------------------------------------
def download_and_extract_scifact() -> None:
    """Download and extract the SciFact dataset if not already present."""
    corpus_path = os.path.join(config.RAW_DATA_DIR, "corpus.jsonl")
    if os.path.exists(corpus_path):
        print("[ingestion] SciFact data already present, skipping download.")
        return

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    tar_path = os.path.join(config.RAW_DATA_DIR, "data.tar.gz")

    print(f"[ingestion] Downloading SciFact from {config.SCIFACT_URL} ...")
    urllib.request.urlretrieve(config.SCIFACT_URL, tar_path)
    print("[ingestion] Download complete. Extracting ...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(config.RAW_DATA_DIR)

    # The tarball extracts to data/ inside RAW_DATA_DIR; move files up if needed
    nested_data_dir = os.path.join(config.RAW_DATA_DIR, "data")
    if os.path.isdir(nested_data_dir):
        for fname in os.listdir(nested_data_dir):
            src = os.path.join(nested_data_dir, fname)
            dst = os.path.join(config.RAW_DATA_DIR, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.rename(src, dst)

    # Clean up tarball
    if os.path.exists(tar_path):
        os.remove(tar_path)

    print("[ingestion] Extraction complete.")


# ---------------------------------------------------------------------------
# 2. Parse JSONL helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 3. Label normalization (Section 3.2 of spec)
# ---------------------------------------------------------------------------
def normalize_label(evidence: dict) -> str:
    """
    Normalize a claim's evidence dict into one of:
    SUPPORTED, REFUTED, NOT_ENOUGH_INFO.
    Returns 'MIXED' for edge-case claims with both SUPPORT & CONTRADICT.
    """
    if not evidence:
        return "NOT_ENOUGH_INFO"

    labels = set()
    for doc_entries in evidence.values():
        for entry in doc_entries:
            labels.add(entry.get("label", "").upper())

    if "SUPPORT" in labels and "CONTRADICT" in labels:
        return "MIXED"  # Will be skipped during sampling
    if "CONTRADICT" in labels:
        return "REFUTED"
    if "SUPPORT" in labels:
        return "SUPPORTED"
    return "NOT_ENOUGH_INFO"


# ---------------------------------------------------------------------------
# 4. Build evaluation set (Section 3.3)
# ---------------------------------------------------------------------------
def build_eval_set() -> List[dict]:
    """Build a balanced 60-claim evaluation set (20 per class, seed=42)."""
    if os.path.exists(config.EVAL_SET_PATH):
        print("[ingestion] eval_set.json already exists, loading.")
        with open(config.EVAL_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # Load dev claims first, supplement with train if needed
    dev_claims = load_jsonl(os.path.join(config.RAW_DATA_DIR, "claims_dev.jsonl"))
    train_claims = load_jsonl(os.path.join(config.RAW_DATA_DIR, "claims_train.jsonl"))

    # Bucket by normalized label
    buckets: Dict[str, List[dict]] = defaultdict(list)
    seen_ids = set()

    for claim in dev_claims:
        label = normalize_label(claim.get("evidence", {}))
        if label == "MIXED":
            continue
        buckets[label].append(claim)
        seen_ids.add(claim["id"])

    # Supplement from train if needed
    for label in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        if len(buckets[label]) < 20:
            for claim in train_claims:
                if claim["id"] in seen_ids:
                    continue
                cl = normalize_label(claim.get("evidence", {}))
                if cl == label:
                    buckets[label].append(claim)
                    seen_ids.add(claim["id"])
                    if len(buckets[label]) >= 20:
                        break

    # Sample 20 per class deterministically
    rng = random.Random(42)
    eval_set = []
    for label in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        pool = buckets[label]
        rng.shuffle(pool)
        selected = pool[:20]
        for c in selected:
            gold_doc_ids = []
            evidence = c.get("evidence", {})
            if evidence:
                gold_doc_ids = [int(did) for did in evidence.keys()]
            eval_set.append({
                "id": c["id"],
                "claim": c["claim"],
                "label": label,
                "gold_doc_ids": gold_doc_ids,
            })

    os.makedirs(os.path.dirname(config.EVAL_SET_PATH), exist_ok=True)
    with open(config.EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"[ingestion] Built eval set: {len(eval_set)} claims "
          f"({sum(1 for c in eval_set if c['label']=='SUPPORTED')} SUP, "
          f"{sum(1 for c in eval_set if c['label']=='REFUTED')} REF, "
          f"{sum(1 for c in eval_set if c['label']=='NOT_ENOUGH_INFO')} NEI)")
    return eval_set


# ---------------------------------------------------------------------------
# 5. Document formatting & chunking
# ---------------------------------------------------------------------------
def format_document(record: dict) -> Tuple[str, str, str]:
    """Format a corpus record into (doc_id, title, full_text)."""
    doc_id = str(record["doc_id"])
    title = record.get("title", "")
    abstract_parts = record.get("abstract", [])
    if isinstance(abstract_parts, list):
        abstract = " ".join(abstract_parts)
    else:
        abstract = str(abstract_parts)
    full_text = f"{title}\n{abstract}"
    return doc_id, title, full_text


def chunk_documents(corpus: List[dict]) -> Tuple[List[str], List[str], List[dict]]:
    """
    Chunk all corpus documents.
    Returns: (chunk_ids, chunk_texts, chunk_metadatas)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_ids = []
    all_texts = []
    all_metas = []

    for record in corpus:
        doc_id, title, full_text = format_document(record)
        chunks = splitter.split_text(full_text)
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_c{idx}"
            all_ids.append(chunk_id)
            all_texts.append(chunk_text)
            all_metas.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
            })

    return all_ids, all_texts, all_metas


# ---------------------------------------------------------------------------
# 6. Tokenizer for BM25
# ---------------------------------------------------------------------------
def tokenize_for_bm25(text: str) -> List[str]:
    """Simple lowercased alphanumeric tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# 7. Build dual indices (Dense ChromaDB + Sparse BM25)
# ---------------------------------------------------------------------------
def build_indices(
    chunk_ids: List[str],
    chunk_texts: List[str],
    chunk_metadatas: List[dict],
) -> None:
    """Build and persist both ChromaDB and BM25 indices."""

    # --- Dense Index (ChromaDB + BGE) ---
    print(f"[ingestion] Loading embedding model: {config.EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"[ingestion] Embedding {len(chunk_texts)} chunks ...")
    embeddings = embedder.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    os.makedirs(config.CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)

    # Delete existing collection if present (for re-indexing)
    try:
        client.delete_collection("scifact_chunks")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="scifact_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB has a batch limit; insert in batches of 5000
    batch_size = 5000
    for i in range(0, len(chunk_ids), batch_size):
        end = min(i + batch_size, len(chunk_ids))
        collection.add(
            ids=chunk_ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=chunk_texts[i:end],
            metadatas=chunk_metadatas[i:end],
        )
    print(f"[ingestion] ChromaDB collection 'scifact_chunks' built with {collection.count()} chunks.")

    # --- Sparse Index (BM25) ---
    print("[ingestion] Building BM25 index ...")
    tokenized_corpus = [tokenize_for_bm25(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "chunk_texts": chunk_texts,
        "chunk_metadatas": chunk_metadatas,
        "tokenized_corpus": tokenized_corpus,
    }
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"[ingestion] BM25 index saved to {config.BM25_INDEX_PATH}")


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------
def run_ingestion() -> None:
    """Full ingestion pipeline: download, parse, chunk, index, eval set."""
    # Step 1: Download & extract
    download_and_extract_scifact()

    # Step 2: Load corpus
    corpus_path = os.path.join(config.RAW_DATA_DIR, "corpus.jsonl")
    print(f"[ingestion] Loading corpus from {corpus_path} ...")
    corpus = load_jsonl(corpus_path)
    print(f"[ingestion] Loaded {len(corpus)} documents.")

    # Step 3: Chunk
    print("[ingestion] Chunking documents ...")
    chunk_ids, chunk_texts, chunk_metas = chunk_documents(corpus)
    print(f"[ingestion] Created {len(chunk_ids)} chunks.")

    # Step 4: Build indices
    build_indices(chunk_ids, chunk_texts, chunk_metas)

    # Step 5: Build eval set
    build_eval_set()

    print("[ingestion] ✅ Ingestion pipeline complete!")


if __name__ == "__main__":
    run_ingestion()
