"""
Offline pre-computation pipeline.
Run once (can take 20-30 min on CPU for 100K candidates).
Produces artifacts that rank.py loads at scoring time.

Usage:
    python precompute.py --candidates ./candidates.jsonl --out ./artifacts/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from features import extract_features
from honeypot import is_honeypot


EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB, 384-dim, fast CPU inference
BATCH_SIZE = 256  # candidates per embedding batch


def load_candidates(jsonl_path: str) -> list[dict]:
    print(f"Loading candidates from {jsonl_path} ...")
    candidates = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates")
    return candidates


def extract_all_features(candidates: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Returns (features_list, candidate_ids) in the same order.
    Also tags each feature with honeypot flag.
    """
    print("Extracting features ...")
    features_list = []
    for cand in tqdm(candidates, desc="Features"):
        feat = extract_features(cand)
        trap, reason = is_honeypot(cand)
        feat["_is_honeypot"] = trap
        feat["_honeypot_reason"] = reason
        features_list.append(feat)
    print(f"Feature extraction complete. Honeypots detected: {sum(1 for f in features_list if f['_is_honeypot'])}")
    return features_list


def build_embeddings(
    features_list: list[dict],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Generate sentence embeddings for all candidates.
    Uses the 'full_text' field from each feature dict.
    Returns float32 array of shape (N, 384).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [feat["full_text"] for feat in features_list]
    print(f"Generating embeddings for {len(texts):,} candidates (batch_size={batch_size}) ...")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine via inner product
    )
    elapsed = time.time() - t0
    print(f"Embeddings done in {elapsed/60:.1f} min. Shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat inner-product index (cosine similarity, since embeddings are L2-normalized).
    For 100K candidates, flat L2 is fast enough (~0.5s per query on CPU).
    """
    print("Building FAISS index ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine for normalized vecs
    index.add(embeddings)
    print(f"FAISS index built. Total vectors: {index.ntotal:,}")
    return index


def save_artifacts(
    out_dir: str,
    features_list: list[dict],
    embeddings: np.ndarray,
    index: faiss.Index,
    candidate_ids: list[str],
) -> None:
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Features — save as JSONL (fast line-by-line load, compact, no pickle overhead)
    feat_path = out / "candidate_features.jsonl"
    print(f"Saving features -> {feat_path}")
    with open(feat_path, "w", encoding="utf-8") as f:
        for feat in features_list:
            # Drop the large full_text field — not needed at rank time
            row = {k: v for k, v in feat.items() if k != "full_text"}
            f.write(json.dumps(row) + "\n")

    # Embeddings
    emb_path = out / "embeddings.npy"
    print(f"Saving embeddings -> {emb_path}")
    np.save(str(emb_path), embeddings)

    # FAISS index
    idx_path = out / "faiss_index.bin"
    print(f"Saving FAISS index -> {idx_path}")
    faiss.write_index(index, str(idx_path))

    # Candidate ID list (for index -> id mapping)
    ids_path = out / "candidate_ids.pkl"
    print(f"Saving candidate IDs -> {ids_path}")
    with open(ids_path, "wb") as f:
        pickle.dump(candidate_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nAll artifacts saved:")
    for p in [feat_path, emb_path, idx_path, ids_path]:
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {p.name}: {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute features and embeddings")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./artifacts", help="Output directory for artifacts")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    t_start = time.time()

    # Load
    candidates = load_candidates(args.candidates)

    # Features
    features_list = extract_all_features(candidates)
    candidate_ids = [f["candidate_id"] for f in features_list]

    # Embeddings
    embeddings = build_embeddings(features_list, batch_size=args.batch_size)

    # FAISS index
    index = build_faiss_index(embeddings)

    # Save
    save_artifacts(args.out, features_list, embeddings, index, candidate_ids)

    total = time.time() - t_start
    print(f"\nPrecomputation complete in {total/60:.1f} minutes.")
    print(f"Artifacts in: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
