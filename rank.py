"""
Main ranking script — must complete in < 5 minutes on CPU, no network.
Loads pre-computed artifacts, scores candidates, outputs top-100 CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv [--artifacts ./artifacts]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from scorer import composite_score
from reasoning import generate_reasoning


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "./artifacts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_N_OUTPUT = 100


# ---------------------------------------------------------------------------
# JD text — embedded at runtime (no network, model is in artifacts/)
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer founding team Redrob AI Series A talent intelligence platform
Pune Noida India hybrid employment full-time 5 to 9 years experience
production embeddings-based retrieval systems sentence-transformers BGE E5 OpenAI embeddings
vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS
strong Python evaluation frameworks ranking systems NDCG MRR MAP A/B testing
LLM fine-tuning LoRA QLoRA PEFT learning to rank XGBoost recommendation system search ranking
NLP information retrieval semantic search dense retrieval RAG retrieval augmented generation
applied machine learning product company startup shipped production real users at scale
ranking retrieval matching intelligence layer candidate job description matching
"""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_artifacts(artifacts_dir: str) -> tuple[list[dict], np.ndarray, list[str]]:
    import json
    d = Path(artifacts_dir)

    print(f"Loading artifacts from {d} ...")
    t0 = time.time()

    feat_path = d / "candidate_features.jsonl"
    features_list: list[dict] = []
    with open(feat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                features_list.append(json.loads(line))

    embeddings: np.ndarray = np.load(str(d / "embeddings.npy"))

    with open(d / "candidate_ids.pkl", "rb") as f:
        candidate_ids: list[str] = pickle.load(f)

    print(f"  Artifacts loaded in {time.time()-t0:.1f}s  ({len(features_list):,} candidates)")
    return features_list, embeddings, candidate_ids


def build_features_map(features_list: list[dict]) -> dict[str, dict]:
    return {f["candidate_id"]: f for f in features_list}


# ---------------------------------------------------------------------------
# JD embedding (in-process, no network — model loaded from local cache)
# ---------------------------------------------------------------------------

def embed_jd(text: str, artifacts_dir: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache_dir = str(Path(artifacts_dir) / "model_cache")
    print("Embedding JD ...")
    t0 = time.time()

    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=cache_dir)
    jd_emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  JD embedded in {time.time()-t0:.1f}s")
    return jd_emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Full-sweep scoring: score all 100K candidates, compute semantic sim via
# dot product (embeddings are L2-normalised so dot = cosine). This gives
# 100% recall — no FAISS pre-filter that could drop valid candidates.
# Timing: ~1.2s score + ~0.1s dot product = well within 5-min limit.
# ---------------------------------------------------------------------------

def score_candidates(
    features_list: list[dict],
    embeddings: np.ndarray,
    candidate_ids: list[str],
    jd_emb: np.ndarray,
) -> list[dict]:
    """
    Score all non-honeypot candidates with real semantic similarity.
    """
    print(f"Computing semantic similarities for all {len(candidate_ids):,} candidates ...")
    t0 = time.time()
    # dot product: (N, 384) @ (384, 1) -> (N,)   O(N·d) ≈ 0.05s
    all_sims = (embeddings @ jd_emb.T).flatten()
    cid_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    print(f"  Similarities computed in {time.time()-t0:.2f}s")

    print(f"Scoring {len(features_list):,} candidates ...")
    t0 = time.time()
    results = []
    for feat in features_list:
        if feat is None:
            continue

        if feat.get("_is_honeypot", False):
            continue

        cid = feat["candidate_id"]
        idx = cid_to_idx.get(cid, -1)
        sem_sim = float(all_sims[idx]) if idx >= 0 else 0.5

        scores = composite_score(feat, semantic_sim=sem_sim)
        results.append({
            "candidate_id": cid,
            "scores": scores,
            "feat": feat,
            "semantic_sim": sem_sim,
        })

    print(f"  Scoring done in {time.time()-t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Selection + tie-break
# ---------------------------------------------------------------------------

def select_top_100(scored: list[dict]) -> list[dict]:
    """
    Sort by final score desc, tie-break by candidate_id asc.
    """
    scored.sort(
        key=lambda x: (-x["scores"]["final"], x["candidate_id"])
    )
    return scored[:TOP_N_OUTPUT]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(top100: list[dict], out_path: str) -> None:
    print(f"Writing submission -> {out_path}")
    rows = []
    for rank, entry in enumerate(top100, start=1):
        cid = entry["candidate_id"]
        feat = entry["feat"]
        scores = entry["scores"]
        reason = generate_reasoning(feat, scores, rank)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": f"{scores['final']:.4f}",
            "reasoning": reason,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows to {out_path}")


# ---------------------------------------------------------------------------
# Validation quick-check
# ---------------------------------------------------------------------------

def quick_validate(out_path: str) -> bool:
    """Lightweight format check before exit."""
    with open(out_path, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    ok = True
    if len(reader) != 100:
        print(f"ERROR: expected 100 rows, got {len(reader)}", file=sys.stderr)
        ok = False

    ranks = [int(r["rank"]) for r in reader]
    if sorted(ranks) != list(range(1, 101)):
        print("ERROR: ranks 1-100 not all present exactly once", file=sys.stderr)
        ok = False

    scores = [float(r["score"]) for r in reader]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1] - 1e-6:
            print(f"ERROR: score not non-increasing at rank {i+1}", file=sys.stderr)
            ok = False
            break

    if ok:
        print("Quick validation: PASSED")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob JD")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--artifacts", default=ARTIFACTS_DIR, help="Artifacts directory")
    args = parser.parse_args()

    wall_start = time.time()

    # 1. Load pre-computed artifacts
    features_list, embeddings, candidate_ids = load_artifacts(args.artifacts)

    # 2. Embed JD (local model, no network)
    jd_emb = embed_jd(JD_TEXT, args.artifacts)

    # 3. Full-sweep: score all candidates with real semantic similarity
    scored = score_candidates(features_list, embeddings, candidate_ids, jd_emb)

    # 4. Select top 100
    top100 = select_top_100(scored)

    # 5. Write CSV
    write_csv(top100, args.out)

    # 6. Quick validate
    quick_validate(args.out)

    wall_elapsed = time.time() - wall_start
    print(f"\nTotal ranking time: {wall_elapsed:.1f}s ({wall_elapsed/60:.2f} min)")
    if wall_elapsed > 300:
        print("WARNING: exceeded 5-minute limit!", file=sys.stderr)


if __name__ == "__main__":
    main()
