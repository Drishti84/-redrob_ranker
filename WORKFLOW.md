# Redrob Ranker — Complete Workflow Plan

## Dataset Diagnostics (from sample of 50 candidates)

| Property | Finding |
|---|---|
| Total candidates | 100,000 |
| Years of experience | 1.1 – 14.5, avg 7.1 |
| Skills per candidate | 5 – 17, avg 9.4 |
| Open to work | ~32% (16/50) |
| Countries | 72% India, 8% USA, rest scattered |
| Notice period | 30/60/90/120/150 days only |
| Education tiers | 50% tier_3, 29% tier_4, 14% tier_2, 7% tier_1 |
| GitHub linked | Only ~34% have GitHub |
| Preferred work mode | ~evenly split remote/hybrid/flexible/onsite |
| AI skill prevalence | LOW — Pinecone, FAISS, RAG etc. are rare in sample |

**Critical insight**: The dataset is dominated by non-AI roles (Business Analyst, Project Manager, Mechanical Engineer, Accountant, etc.). The AI/ML candidates are a small minority within 100K. This confirms the JD's own warning — "We'd rather see 10 great matches than 1000 maybes."

---

## Architecture Decision

### Why NOT per-candidate LLM calls:
- 100K candidates × ~1s per LLM call = 27+ hours → violates 5-min constraint
- Even a local 7B model: ~100K × 0.5s = 14 hours → still impossible

### Chosen Architecture: Two-Phase Hybrid Scorer

**Phase 1: Offline Pre-computation** (no time limit, run once)
- Parse all 100K candidates
- Extract structured features per candidate
- Generate text embeddings for full profiles using a lightweight model
- Build FAISS index for fast retrieval
- Generate JD embedding
- Save all features + index to disk

**Phase 2: Online Ranking** (< 5 min, CPU only, no network)
- Load pre-computed features + FAISS index
- ANN search: retrieve top 5K–10K candidates by embedding similarity
- Apply weighted rule-based scoring on those 5K–10K
- Honeypot detection pass
- Output top 100 with generated reasoning

---

## Scoring Formula (Weighted Multi-Signal)

```
final_score = (
    0.35 * skill_match_score     # Core JD skill coverage + depth
  + 0.25 * career_quality_score  # Product company AI/ML experience
  + 0.20 * availability_score    # Behavioral signals: can we actually hire them
  + 0.15 * semantic_fit_score    # Embedding cosine similarity vs JD
  + 0.05 * education_score       # Institution tier bonus
)
```

### Component: `skill_match_score` (0–1)

JD requires (hard):
```python
REQUIRED_SKILLS = [
    # embeddings / retrieval
    "sentence-transformers", "embeddings", "vector search", "semantic search",
    "retrieval", "RAG", "dense retrieval",
    # vector DBs
    "faiss", "pinecone", "milvus", "weaviate", "qdrant", "opensearch",
    "elasticsearch", "vector database",
    # evaluation
    "NDCG", "MRR", "MAP", "ranking evaluation", "A/B testing",
    # core
    "python", "NLP", "information retrieval",
]
NICE_TO_HAVE = [
    "LoRA", "QLoRA", "fine-tuning", "PEFT", "LLM",
    "XGBoost", "learning to rank", "LTR",
    "recommendation system", "search ranking",
    "distributed systems", "inference optimization",
]
```

**Scoring logic**:
- Check each skill by name (fuzzy match) + check career history descriptions for keywords
- Weight each skill by: `proficiency_weight * endorsement_boost * duration_weight`
  - proficiency: beginner=0.3, intermediate=0.6, advanced=0.85, expert=1.0
  - endorsement_boost: log(1 + endorsements) normalized
  - duration: months / 24 capped at 1.0 (2+ years = full credit)
- Skill assessment scores (from redrob_signals) act as truth multiplier on claimed proficiency
- Required skills coverage = 0..1, nice-to-have adds bonus up to 0.15

### Component: `career_quality_score` (0–1)

Positive signals:
- Roles containing: "AI Engineer", "ML Engineer", "Data Scientist", "Research Scientist", "NLP", "Search", "Ranking", "Recommendation", "Retrieval"
- Industry: "Technology", "AI", "SaaS", "Product", "Fintech", "E-commerce" (not "IT Services" from pure consulting)
- Company tenure diversity: worked at product companies (not ONLY TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini)
- Has shipped actual systems (keywords in description: "deployed", "shipped", "production", "users", "latency", "scale")

Penalties:
- ALL career entries are at: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL → heavy penalty (-0.4)
- ALL roles are "researcher" / "lab" / "academic" → heavy penalty (-0.3)
- Average company tenure < 18 months (title-chaser pattern) → penalty (-0.2)
- Current title is radically off (Marketing Manager, Accountant, Mechanical Engineer) → penalty (-0.5)

### Component: `availability_score` (0–1)

```python
def availability_score(signals):
    score = 0.0
    # Open to work flag (binary boost)
    if signals['open_to_work_flag']:
        score += 0.20
    # Recency of activity
    days_since_active = (today - signals['last_active_date']).days
    score += max(0, 0.20 * (1 - days_since_active / 180))
    # Recruiter response rate
    score += 0.20 * signals['recruiter_response_rate']
    # Interview completion
    score += 0.15 * signals['interview_completion_rate']
    # Notice period (sub-30 = great, 30-60 = good, 90+ = worse)
    notice_score = max(0, 1 - (signals['notice_period_days'] - 30) / 120)
    score += 0.10 * notice_score
    # Willing to relocate to Pune/Noida
    if signals['willing_to_relocate']:
        score += 0.08
    # Profile completeness
    score += 0.07 * (signals['profile_completeness_score'] / 100)
    return min(1.0, score)
```

### Component: `semantic_fit_score` (0–1)

- Embed JD text using `all-MiniLM-L6-v2` (22MB, fast, CPU-friendly)
- Embed each candidate's combined text: headline + summary + career descriptions
- Cosine similarity → normalize to 0–1
- Used for FAISS first-pass filtering and as signal in final scoring

### Component: `education_score` (0–1)

```python
TIER_SCORES = {'tier_1': 1.0, 'tier_2': 0.75, 'tier_3': 0.45, 'tier_4': 0.2, 'unknown': 0.3}
# Take max tier across all education entries
```

---

## Honeypot Detection

Check these impossible patterns and assign score = 0:

```python
def is_honeypot(candidate):
    for job in candidate['career_history']:
        start = parse_date(job['start_date'])
        end = parse_date(job['end_date']) if job['end_date'] else today
        actual_months = (end - start).days / 30.44
        # Claimed duration wildly different from date math
        if abs(actual_months - job['duration_months']) > 12:
            return True
    
    # Expert skill with 0 months used
    for skill in candidate['skills']:
        if skill['proficiency'] == 'expert' and skill.get('duration_months', 0) == 0:
            return True
    
    # Years of experience exceeds career history span
    # (catch: 10 YoE but all jobs only add up to 3 years)
    total_career_months = sum(j['duration_months'] for j in candidate['career_history'])
    if candidate['profile']['years_of_experience'] > (total_career_months / 12) + 3:
        return True
    
    return False
```

---

## File Structure

```
redrob_ranker/
├── RULES_AND_CONSTRAINTS.md     ← Challenge rules (this doc's sibling)
├── WORKFLOW.md                  ← This file
├── precompute.py                ← Offline: extract features, build embeddings + FAISS index
├── rank.py                      ← Online: load index, score, output CSV (must be < 5 min)
├── features.py                  ← Feature extraction functions
├── scorer.py                    ← Scoring logic (skill_match, career_quality, availability)
├── honeypot.py                  ← Honeypot detection
├── reasoning.py                 ← Generate per-candidate reasoning strings
├── requirements.txt             ← All dependencies + versions
├── submission_metadata.yaml     ← Filled submission metadata
├── README.md                    ← Setup + exact reproduce command
└── artifacts/                   ← Pre-computed (gitignore the large ones)
    ├── candidate_features.pkl   ← Structured features for all 100K
    ├── embeddings.npy           ← Candidate embeddings (100K × 384)
    └── faiss_index.bin          ← FAISS flat index
```

---

## Step-by-Step Implementation Plan

### Step 1: `features.py` — Feature Extraction
Extract structured features from each raw candidate JSON:
- `yoe` — years of experience
- `current_title_category` — normalized (AI_ENGINEER / ML_ENGINEER / DATA_SCIENTIST / SOFTWARE_ENGINEER / NON_AI / ...)
- `company_type_history` — list: PRODUCT / CONSULTING / RESEARCH / UNKNOWN per role
- `ai_skill_count` — number of JD-relevant skills
- `skill_match_details` — dict of matched skill names + weights
- `avg_tenure_months` — average company tenure
- `consulting_only_flag` — bool
- `has_production_deployment` — bool (from career description keywords)
- `all text` — concatenated profile text for embedding

### Step 2: `precompute.py` — Offline Processing (run once)
```bash
python precompute.py --candidates ./candidates.jsonl --out ./artifacts/
```
- Loads all 100K candidates (~464 MB)
- Extracts features for all
- Generates embeddings (batch, ~15-20 min on CPU)
- Builds FAISS index
- Saves artifacts

### Step 3: `scorer.py` — Scoring Engine
Pure functions, no I/O. Takes pre-computed features, returns scores.

### Step 4: `honeypot.py` — Trap Detection
Returns bool per candidate.

### Step 5: `reasoning.py` — Reasoning Generator
Takes top-100 candidates + their scores + component scores, generates specific 1-2 sentence reasoning using templates filled with actual facts from the profile. NOT templated — uses branching logic based on what makes each candidate distinctive.

### Step 6: `rank.py` — Main Ranking Script (< 5 min)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
1. Load artifacts (embeddings, features, FAISS index) — ~30 seconds
2. Compute JD embedding (in-memory, no network)
3. FAISS search: retrieve top 10K by cosine similarity — ~5 seconds
4. Score those 10K using scorer.py — ~60 seconds
5. Honeypot filter
6. Sort, pick top 100
7. Generate reasoning for each
8. Write CSV

Total estimated time: ~2–3 minutes on 16GB CPU machine.

---

## Key Libraries

| Library | Purpose | Size |
|---|---|---|
| `sentence-transformers` | Embeddings (MiniLM) | ~120MB |
| `faiss-cpu` | ANN search | ~50MB |
| `numpy` | Fast array ops | standard |
| `pandas` | CSV/JSONL handling | standard |
| `scikit-learn` | Normalization | standard |
| `rapidfuzz` | Fuzzy skill matching | small |

---

## Scoring Weights Rationale

- **Skill match (35%)**: The JD lists very specific required technical skills. A candidate without production embeddings experience simply cannot do this job.
- **Career quality (25%)**: The JD explicitly says they've had bad experiences with consulting-only and research-only profiles. Career history is the best truth-signal.
- **Availability (20%)**: "A perfect-on-paper candidate who hasn't logged in for 6 months... is not actually available." Behavioral signals are a hard constraint.
- **Semantic fit (15%)**: Catches candidates whose *descriptions* match even if their skill list doesn't use exact JD keywords (the "Tier 5" mentioned in JD notes).
- **Education (5%)**: Modest signal — JD doesn't mention education tier explicitly, but tier_1 calibrates seniority expectations.

---

## Edge Cases to Handle

1. `github_activity_score = -1` → treat as 0, don't penalize (many strong engineers don't public-GitHub)
2. `offer_acceptance_rate = -1` → exclude from scoring (no history, neutral)
3. `end_date = null` for current job → use today's date for duration calculation
4. Empty `certifications` or `languages` → fine, just skip
5. `skill_assessment_scores` may not have all skills → only use what's present
6. Candidates outside India → not automatic disqualifier if `willing_to_relocate = true`

---

## Reasoning Template Logic

```python
def generate_reasoning(candidate, scores, rank):
    facts = []
    
    # Lead with most impressive qualifier
    if scores['career_quality'] > 0.7:
        facts.append(f"{yoe} yrs applied ML at product companies")
    elif scores['skill_match'] > 0.7:
        facts.append(f"Strong skills: {top_matched_skills}")
    
    # Add specific signal
    if signals['open_to_work_flag'] and notice_period <= 30:
        facts.append(f"open to work, {notice_period}d notice")
    elif not signals['open_to_work_flag']:
        facts.append("not marked open to work — may need outreach")
    
    # Honest concern for lower ranks
    if rank > 50:
        if consulting_only:
            facts.append("career history is entirely services/consulting")
        if days_since_active > 90:
            facts.append(f"last active {days_since_active}d ago")
    
    return "; ".join(facts) + "."
```

---

## Validation Checklist (before submitting)

- [ ] Run `python validate_submission.py submission.csv` → 0 errors
- [ ] Confirm exactly 100 data rows
- [ ] Confirm ranks 1–100 each appear exactly once
- [ ] Confirm score is non-increasing (check adjacent rows)
- [ ] Confirm no duplicate candidate_ids
- [ ] Confirm all candidate_ids exist in candidates.jsonl
- [ ] Manually inspect top 10 — do they make sense?
- [ ] Check if any honeypots snuck into top 100
- [ ] Reasoning column is non-empty for all 100
- [ ] Reasoning cites specific facts (not generic praise)
- [ ] Tie scores: verify candidate_id ascending order
- [ ] Run timing: `time python rank.py --candidates ./candidates.jsonl --out ./test.csv` → < 5 min
