# Redrob Hackathon — Rules, Constraints & Notes

**Challenge**: Intelligent Candidate Discovery & Ranking  
**Dataset**: 100,000 candidate profiles (`candidates.jsonl`, ~464 MB)  
**Task**: Rank the top 100 candidates for the given job description

---

## THE JOB DESCRIPTION (what we're ranking for)

**Role**: Senior AI Engineer — Founding Team at Redrob AI (Series A)  
**Location**: Pune/Noida (Hybrid) — open to Tier-1 Indian cities (Hyd, Pune, Mumbai, Delhi NCR)  
**Experience**: 5–9 years (flexible if signals are strong)  
**Employment**: Full-time

### Hard Requirements (MUST HAVE)
1. Production experience with **embeddings-based retrieval** (sentence-transformers, BGE, E5, OpenAI embeddings, etc.) — deployed to real users
2. Production experience with **vector databases / hybrid search** (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, FAISS, Elasticsearch, etc.)
3. Strong **Python** — code quality matters
4. Experience designing **evaluation frameworks for ranking** (NDCG, MRR, MAP, A/B tests, offline-to-online correlation)

### Nice-to-Have (won't reject without)
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank (XGBoost-based or neural)
- HR-tech / marketplace product background
- Distributed systems / large-scale inference
- Open-source AI/ML contributions

### EXPLICIT DISQUALIFIERS (must penalize)
- **Pure research only** — no production deployment (academic labs, research-only roles)
- **LangChain-only AI** — less than 12 months of LLM-era experience without pre-LLM ML production background
- **Not coding** — senior engineers who haven't written production code in last 18 months (moved to pure "architecture" role)
- **Consulting-only career** — TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini — if their ENTIRE career is services firms (note: OK if they have prior product company experience)
- **Title-chasers** — switching companies every 1.5 years to climb titles (avg tenure < 1.5 yrs)
- **Framework enthusiasts without depth** — only LangChain tutorials, not systems thinking
- **CV/Speech/Robotics only** — without significant NLP/IR exposure

### Ideal Candidate Profile
- 6–8 years total; 4–5 years in **applied ML at product companies** (not services)
- Shipped at least **one end-to-end ranking/search/recommendation system** to real users at scale
- Strong opinions on retrieval (hybrid vs dense), evaluation, LLM integration — and can back them with shipped systems
- Located in or willing to relocate to Noida/Pune
- **Active on platform** — in the job market right now
- **Sub-30 day notice period** preferred (they can buy out up to 30 days; 30+ day notice = higher bar)

---

## SUBMISSION FORMAT RULES

### File Format
- **CSV only** — UTF-8 encoded, `.csv` extension
- **Filename**: your team's registered participant ID (e.g., `team_xxx.csv`)
- **Exactly 4 columns in this exact order**: `candidate_id,rank,score,reasoning`

### Row Rules
- **Exactly 100 data rows** (+ 1 header row = 101 total lines)
- Each **rank 1–100 appears exactly once**
- Each **candidate_id appears exactly once**
- Every `candidate_id` must **exist in candidates.jsonl**
- `score` must be **non-increasing** with rank (rank 1 has highest score)
- **Tie-break**: same score → sort by `candidate_id` ascending
- `rank` must be an **integer** (not float, not starting at 0)

### Reasoning Column
- Optional but **heavily recommended** — affects Stage 4 manual review
- 1–2 sentences per candidate
- Must reference **specific facts** from the profile (years, title, named skills, signal values)
- Must connect to **specific JD requirements**
- Must be **honest** — acknowledge gaps where they exist
- **No hallucination** — never mention skills/companies not in the profile
- Must be **varied** — not templated
- Tone must **match the rank** (rank-5 shouldn't sound worse than rank-95)

---

## COMPUTE CONSTRAINTS (ranking step only)

| Constraint | Limit |
|---|---|
| Total runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — NO GPU |
| Network | OFF — no API calls (no OpenAI, Anthropic, Cohere, Gemini, etc.) |
| Disk | ≤ 5 GB intermediate state |

- **Pre-computation is allowed** (generating embeddings, indexes, model weights) — can take longer
- Only the **ranking step** (the code that produces the CSV) must be < 5 min
- The ranking step is reproduced in a **sandboxed Docker container** at Stage 3

---

## SCORING METRICS

| Metric | Weight | What it measures |
|---|---|---|
| NDCG@10 | **50%** | Quality of your top-10 picks (MOST IMPORTANT) |
| NDCG@50 | 30% | Quality of your top-50 picks |
| MAP | 15% | Precision across all relevance levels |
| P@10 | 5% | Fraction of top-10 that are "relevant" (tier 3+) |

**Final composite** = `0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

**Tiebreaks**: Higher P@5 → Higher P@10 → Earlier submission timestamp

---

## HONEYPOT WARNING (CRITICAL)

- Dataset contains **~80 honeypot candidates** with subtly impossible profiles
- Examples: 8 years experience at a company founded 3 years ago; "expert" proficiency in 10 skills with 0 months used
- These are forced to **relevance tier 0** in ground truth
- **Honeypot rate > 10% in top 100 = DISQUALIFICATION at Stage 3**
- Don't special-case them — a good ranking system naturally avoids them
- Detection: check career date math, check skill proficiency vs. duration_months

### Other Traps in Dataset
- **Keyword stuffers** — Marketing Managers with all AI skills listed = NOT a fit
- **Plain-language Tier 5s** — no AI keywords but career history proves fit
- **Behavioral twins** — same skill match but wildly different availability signals

---

## EVALUATION STAGES

| Stage | What happens | What eliminates you |
|---|---|---|
| 1 | Format validation (auto) | Any spec violation |
| 2 | Scoring against hidden ground truth | Score below cutoff |
| 3 | Code reproduction + honeypot check | Can't reproduce; honeypot rate > 10%; fake/missing repo |
| 4 | Manual review — reasoning quality | Failed reasoning checks; flat git history; LLM-only codebase |
| 5 | Defend-your-work interview | Can't explain your architecture |

---

## SUBMISSION PACKAGE (3 parts, all required)

1. **CSV file** — top-100 ranking
2. **Portal metadata** — team name, contact, GitHub repo URL, sandbox link, AI tools declared, compute env, team members
3. **GitHub repo** — clean code, README with single reproduce command, `requirements.txt`, `submission_metadata.yaml`

### Sandbox
- A working hosted environment where the ranker can run on a small sample
- Valid: HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, Binder

### Reproduce Command
```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## THREE SUBMISSION CAP

- Maximum **3 submissions** — last valid one counts
- No live leaderboard — no per-submission feedback during competition
- Validate locally with `validate_submission.py` before submitting

---

## KEY INSIGHTS FROM JD (for scoring)

> "The right answer involves reasoning about the gap between what the JD says and what the JD means."
> "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' but if their career history shows they built a recommendation system at a product company, they're a fit."
> "A candidate who has all AI keywords as skills but whose title is 'Marketing Manager' is not a fit."
> "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."
> "We'd rather see 10 great matches than 1000 maybes."
