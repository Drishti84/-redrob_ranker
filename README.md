# Redrob Ranker — Senior AI Engineer Candidate Ranking System

Ranks the top 100 candidates from a 100K-candidate pool for the Redrob Senior AI Engineer JD.

## Architecture

Two-phase pipeline:

1. **Precompute** (run once, ~20-30 min, can use network):
   - Extracts structured features from all 100K candidate profiles
   - Generates sentence embeddings using `all-MiniLM-L6-v2`
   - Builds a FAISS flat-IP index for fast approximate nearest-neighbor search
   - Saves all artifacts to disk

2. **Rank** (< 5 min, CPU only, no network):
   - Loads pre-computed artifacts
   - Embeds the JD using the locally-cached model
   - Retrieves top 10K candidates via FAISS cosine similarity
   - Scores all 10K with a weighted 5-component scorer
   - Outputs top 100 with fact-grounded reasoning

### Scoring weights

| Component | Weight | What it captures |
|---|---|---|
| Skill match | 35% | JD-required skills (embeddings, vector DBs, ranking eval, NLP) |
| Career quality | 25% | Product-company AI/ML history; penalizes consulting-only and wrong-domain titles |
| Availability | 20% | Behavioral signals: open-to-work, last active, response rate, notice period |
| Semantic fit | 15% | Embedding cosine similarity — catches implied skills from career descriptions |
| Education | 5% | Institution tier bonus |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1 — Pre-computation (run once)

```bash
python precompute.py --candidates ./candidates.jsonl --out ./artifacts/
```

This downloads the embedding model on first run, then processes all 100K candidates.
Artifacts are saved to `./artifacts/` (~600 MB total).

### Step 2 — Ranking (the submission-time command)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This is the exact reproduce command. Must complete in < 5 minutes on CPU.

### Validate before submitting

```bash
python validate_submission.py submission.csv
```

## Project structure

```
redrob_ranker/
├── rank.py                    ← Main ranking script (run this for submission)
├── precompute.py              ← Offline preprocessing (run once)
├── features.py                ← Feature extraction from raw candidate JSON
├── scorer.py                  ← Weighted multi-signal scoring engine
├── honeypot.py                ← Honeypot detection (impossible profiles)
├── reasoning.py               ← Per-candidate reasoning string generator
├── requirements.txt           ← Pinned dependencies
├── submission_metadata.yaml   ← Team metadata
├── RULES_AND_CONSTRAINTS.md   ← Full challenge rules reference
├── WORKFLOW.md                ← Architecture decisions and scoring rationale
└── artifacts/                 ← Pre-computed artifacts (not committed to git)
    ├── candidate_features.pkl
    ├── embeddings.npy
    ├── faiss_index.bin
    ├── candidate_ids.pkl
    └── model_cache/           ← Local copy of all-MiniLM-L6-v2
```

## Honeypot handling

The dataset contains ~80 synthetic trap candidates with impossible profiles
(e.g., 8 years at a 3-year-old company; expert skills with 0 months used).
`honeypot.py` detects these at precompute time and filters them from scoring.

## Key design decisions

- **No per-candidate LLM calls** — 100K × LLM call = hours, violates 5-min constraint.
- **Semantic search is a first-pass filter**, not the final ranking signal — pure embedding similarity rewards keyword stuffers.
- **Career quality score is a hard gate** — a Marketing Manager with every AI keyword still scores low because career history is evaluated independently.
- **Behavioral availability is 20% of the score** — an unavailable perfect-on-paper candidate is useless to a recruiter.
