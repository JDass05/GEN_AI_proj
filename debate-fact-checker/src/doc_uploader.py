"""Document ingestion utility for dynamically uploading and adding user files
(PDF, DOCX, TXT, Markdown) into the active ChromaDB and BM25 index.
"""

import os
import pickle
import re
import uuid
from typing import List, Tuple

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from src.ingestion import tokenize_for_bm25


def parse_file_content(file_path: str) -> Tuple[str, str]:
    """Extract plain text and title from a PDF, DOCX, TXT, or MD file."""
    ext = os.path.splitext(file_path)[1].lower()
    title = os.path.basename(file_path)
    text = ""

    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages_text = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages_text.append(t)
        text = "\n\n".join(pages_text)

    elif ext in [".docx", ".doc"]:
        import docx
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)

    elif ext in [".txt", ".md", ".json", ".csv"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        # Generic fallback
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return title, text.strip()


def add_document_to_index(file_path: str) -> dict:
    """
    Chunks a new document, embeds it using BGE on the active device (GPU if available),
    inserts it into ChromaDB, and updates the persisted BM25 index.
    """
    title, text = parse_file_content(file_path)
    if not text:
        return {"status": "error", "message": "File is empty or could not be read."}

    doc_id = f"user_{uuid.uuid4().hex[:8]}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_text(text)
    if not chunks:
        return {"status": "error", "message": "No text chunks generated from file."}

    chunk_ids = []
    chunk_texts = []
    chunk_metas = []

    for idx, c_text in enumerate(chunks):
        cid = f"{doc_id}_c{idx}"
        chunk_ids.append(cid)
        chunk_texts.append(c_text)
        chunk_metas.append({
            "chunk_id": cid,
            "doc_id": doc_id,
            "title": title,
            "source": "user_upload",
        })

    # 1. Embed and add to ChromaDB
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    embeddings = embedder.encode(
        chunk_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )

    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name="scifact_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings.tolist(),
        documents=chunk_texts,
        metadatas=chunk_metas,
    )

    # 2. Update BM25 index
    bm25_data = None
    if os.path.exists(config.BM25_INDEX_PATH):
        try:
            with open(config.BM25_INDEX_PATH, "rb") as f:
                bm25_data = pickle.load(f)
        except Exception:
            pass

    if bm25_data is not None:
        bm25_data["chunk_ids"].extend(chunk_ids)
        bm25_data["chunk_texts"].extend(chunk_texts)
        bm25_data["chunk_metadatas"].extend(chunk_metas)
        new_tokens = [tokenize_for_bm25(t) for t in chunk_texts]
        bm25_data["tokenized_corpus"].extend(new_tokens)
        bm25_data["bm25"] = BM25Okapi(bm25_data["tokenized_corpus"])
    else:
        tokenized_corpus = [tokenize_for_bm25(t) for t in chunk_texts]
        bm25_data = {
            "bm25": BM25Okapi(tokenized_corpus),
            "chunk_ids": chunk_ids,
            "chunk_texts": chunk_texts,
            "chunk_metadatas": chunk_metas,
            "tokenized_corpus": tokenized_corpus,
        }

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    # Reset any lazy-loaded retriever singletons if module is imported
    try:
        import src.retriever as retriever_module
        retriever_module._chroma_collection = None
        retriever_module._bm25_data = None
    except Exception:
        pass

    return {
        "status": "success",
        "doc_id": doc_id,
        "title": title,
        "chunks_added": len(chunk_ids),
        "total_corpus_chunks": collection.count(),
    }
