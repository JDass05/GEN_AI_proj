"""
Hybrid retrieval pipeline: Dense (ChromaDB/BGE) + Sparse (BM25) + RRF + Cross-Encoder reranking.

Usage:
    from src.retriever import retrieve_evidence
    chunks = retrieve_evidence("Some claim to fact-check", top_k=5)
"""

import os
import pickle
import re
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

import config
from src.schemas import EvidenceChunk


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_embedder: SentenceTransformer = None
_cross_encoder: CrossEncoder = None
_chroma_collection = None
_bm25_data: dict = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[retriever] Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _embedder


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[retriever] Loading cross-encoder: {config.RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(config.RERANKER_MODEL, device="cpu")
    return _cross_encoder


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        _chroma_collection = client.get_collection("scifact_chunks")
        print(f"[retriever] Loaded ChromaDB collection with {_chroma_collection.count()} chunks.")
    return _chroma_collection


def _get_bm25_data() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
        print(f"[retriever] Loaded BM25 index with {len(_bm25_data['chunk_ids'])} chunks.")
    return _bm25_data


# ---------------------------------------------------------------------------
# Tokenizer (must match ingestion)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Dense search
# ---------------------------------------------------------------------------
def _dense_search(claim: str, top_n: int = None) -> Dict[str, float]:
    """
    Query ChromaDB with claim embedding. Returns {chunk_id: distance_score}.
    BGE queries should be prefixed for best performance.
    """
    top_n = top_n or config.DENSE_TOP_N
    embedder = _get_embedder()
    query_text = f"Represent this sentence for searching relevant passages: {claim}"
    query_embedding = embedder.encode(query_text, normalize_embeddings=True).tolist()

    collection = _get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    scores = {}
    if results and results["ids"] and results["ids"][0]:
        for cid, dist in zip(results["ids"][0], results["distances"][0]):
            # ChromaDB cosine distance: lower is better. Convert to similarity.
            scores[cid] = 1.0 - dist
    return scores


# ---------------------------------------------------------------------------
# Sparse search (BM25)
# ---------------------------------------------------------------------------
def _bm25_search(claim: str, top_n: int = None) -> Dict[str, float]:
    """Query BM25 index. Returns {chunk_id: bm25_score}."""
    top_n = top_n or config.BM25_TOP_N
    bm25_data = _get_bm25_data()
    bm25 = bm25_data["bm25"]
    chunk_ids = bm25_data["chunk_ids"]

    tokenized_query = _tokenize(claim)
    raw_scores = bm25.get_scores(tokenized_query)

    # Get top-n indices
    scored = [(i, raw_scores[i]) for i in range(len(raw_scores))]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = scored[:top_n]

    return {chunk_ids[i]: score for i, score in top_indices if score > 0}


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(
    dense_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    k: int = None,
    top_n: int = None,
) -> List[str]:
    """
    Merge dense and BM25 ranked lists using RRF.
    Returns sorted list of chunk_ids (best first).
    """
    k = k or config.RRF_K
    top_n = top_n or config.RRF_TOP_N

    # Build ranked lists (sorted by score descending)
    dense_ranked = sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True)
    bm25_ranked = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)

    # Compute rank positions (1-indexed)
    dense_rank = {cid: rank + 1 for rank, cid in enumerate(dense_ranked)}
    bm25_rank = {cid: rank + 1 for rank, cid in enumerate(bm25_ranked)}

    # All candidate chunk IDs
    all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())

    rrf_scores = {}
    for cid in all_ids:
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (k + dense_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (k + bm25_rank[cid])
        rrf_scores[cid] = score

    # Sort by RRF score descending and take top_n
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return sorted_ids[:top_n]


# ---------------------------------------------------------------------------
# Cross-Encoder Reranking
# ---------------------------------------------------------------------------
def _rerank(claim: str, candidate_ids: List[str], top_k: int) -> List[EvidenceChunk]:
    """Score (claim, chunk) pairs with cross-encoder and return top_k EvidenceChunks."""
    bm25_data = _get_bm25_data()
    chunk_ids_all = bm25_data["chunk_ids"]
    chunk_texts_all = bm25_data["chunk_texts"]
    chunk_metas_all = bm25_data["chunk_metadatas"]

    # Build lookup
    id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids_all)}

    # Prepare pairs
    pairs = []
    valid_ids = []
    for cid in candidate_ids:
        if cid in id_to_idx:
            idx = id_to_idx[cid]
            pairs.append((claim, chunk_texts_all[idx]))
            valid_ids.append(cid)

    if not pairs:
        return []

    cross_encoder = _get_cross_encoder()
    scores = cross_encoder.predict(pairs)

    # Combine and sort
    scored_chunks = []
    for cid, score in zip(valid_ids, scores):
        idx = id_to_idx[cid]
        meta = chunk_metas_all[idx]
        chunk = EvidenceChunk(
            chunk_id=cid,
            doc_id=meta["doc_id"],
            title=meta["title"],
            text=chunk_texts_all[idx],
            rerank_score=float(score),
        )
        scored_chunks.append(chunk)

    scored_chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    return scored_chunks[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def retrieve_evidence(claim: str, top_k: int = None) -> List[EvidenceChunk]:
    """
    Full hybrid retrieval pipeline:
    1. Dense search (ChromaDB + BGE)
    2. Sparse search (BM25)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Reranking
    Returns the top_k most relevant EvidenceChunks.
    """
    top_k = top_k or config.TOP_K_EVIDENCE

    # Step 1 & 2: Dual retrieval
    dense_scores = _dense_search(claim)
    bm25_scores = _bm25_search(claim)

    # Step 3: RRF fusion
    fused_candidates = _reciprocal_rank_fusion(dense_scores, bm25_scores)

    # Step 4: Cross-encoder reranking
    top_chunks = _rerank(claim, fused_candidates, top_k)

    return top_chunks
