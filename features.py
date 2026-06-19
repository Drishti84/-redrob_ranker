"""
Feature extraction from raw candidate JSON.
All functions are pure and stateless — no I/O.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# JD-derived skill taxonomy
# ---------------------------------------------------------------------------

# Core required skills — must have to do this job
REQUIRED_SKILLS: list[str] = [
    # Embeddings / retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "embedding",
    "vector search", "semantic search", "dense retrieval", "rag", "retrieval augmented",
    "bi-encoder", "cross-encoder", "passage retrieval", "neural search",
    # Vector DBs
    "faiss", "pinecone", "milvus", "weaviate", "qdrant", "opensearch",
    "elasticsearch", "vector database", "vector db", "annoy", "hnsw",
    # Ranking / retrieval evaluation
    "ndcg", "mrr", "map", "mean average precision", "ranking evaluation",
    "information retrieval", "ir", "bm25", "hybrid search", "hybrid retrieval",
    # NLP
    "nlp", "natural language processing", "text classification", "named entity",
    "transformers", "bert", "roberta", "llm", "large language model",
    "text embeddings", "word2vec", "doc2vec",
    # Python (inferred from title/career — listed separately)
    "python",
]

# Nice-to-have skills — bonus but not required
NICE_TO_HAVE_SKILLS: list[str] = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "learning to rank", "ltr", "lambdamart", "ranknet",
    "recommendation", "recommender", "collaborative filtering",
    "distributed", "inference optimization", "model serving", "triton",
    "mlflow", "weights & biases", "wandb", "mlops",
    "spark", "kafka", "airflow", "data pipeline",
    "pytorch", "tensorflow", "jax",
    "a/b testing", "experiment", "online evaluation",
    "reranking", "reranker", "cross-encoder reranking",
    "rag pipeline", "langchain", "llamaindex", "haystack",
]

# Hard negative: title is wrong domain entirely
NON_AI_TITLES: set[str] = {
    "accountant", "civil engineer", "mechanical engineer", "graphic designer",
    "content writer", "hr manager", "sales executive", "marketing manager",
    "operations manager", "project manager", "business analyst",
    "customer support", "customer service", "frontend engineer",
    "mobile developer", ".net developer", "java developer",
    "ui designer", "ux designer", "product designer",
}

# AI / ML related titles — positive signal
AI_TITLES: set[str] = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "research scientist", "applied scientist",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "retrieval engineer",
    "data engineer", "software engineer", "backend engineer",
    "full stack", "fullstack", "senior engineer", "staff engineer",
    "principal engineer", "tech lead", "engineering manager",
    "junior ml engineer", "senior machine learning engineer",
}

# Consulting companies — signal for services-only career
CONSULTING_FIRMS: set[str] = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra",
    "mphasis", "hexaware", "l&t infotech", "ltimindtree",
    "mindtree", "niit technologies",
}

# Proficiency weights
PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}

# Education tier scores
TIER_SCORES: dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.45,
    "tier_4": 0.2,
    "unknown": 0.3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _normalize(text: str) -> str:
    return text.lower().strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    t = _normalize(text)
    return any(term in t for term in terms)


def _skill_name_matches(skill_name: str, terms: list[str]) -> bool:
    n = _normalize(skill_name)
    return any(term in n or n in term for term in terms)


def _is_consulting(company: str) -> bool:
    c = _normalize(company)
    return any(firm in c for firm in CONSULTING_FIRMS)


def _is_ai_title(title: str) -> bool:
    t = _normalize(title)
    return any(at in t for at in AI_TITLES)


def _is_non_ai_title(title: str) -> bool:
    t = _normalize(title)
    return any(nat in t for nat in NON_AI_TITLES)


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def extract_features(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Return a flat feature dict for one candidate.
    All downstream scoring uses only this dict — not the raw candidate.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})

    cid = candidate["candidate_id"]
    yoe = float(profile.get("years_of_experience", 0))
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    country = profile.get("country", "")
    location = profile.get("location", "")
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    # ------------------------------------------------------------------
    # Skills analysis
    # ------------------------------------------------------------------
    required_hits: list[dict] = []
    nice_hits: list[dict] = []
    skill_names: list[str] = []

    for sk in skills:
        name = sk.get("name", "")
        skill_names.append(name)
        prof = sk.get("proficiency", "intermediate")
        duration = sk.get("duration_months", 0) or 0
        endorsements = sk.get("endorsements", 0) or 0

        prof_w = PROFICIENCY_WEIGHTS.get(prof, 0.6)
        dur_w = min(1.0, duration / 24.0)
        end_w = min(1.0, (endorsements / 20.0) ** 0.5)
        raw_w = prof_w * (0.6 + 0.4 * dur_w) * (0.7 + 0.3 * end_w)

        # Apply assessment score multiplier if available
        assessment_scores = signals.get("skill_assessment_scores", {})
        if name in assessment_scores:
            assessed = assessment_scores[name] / 100.0
            raw_w = raw_w * (0.5 + 0.5 * assessed)

        entry = {"name": name, "weight": round(raw_w, 4), "duration": duration}

        if _skill_name_matches(name, REQUIRED_SKILLS):
            required_hits.append(entry)
        elif _skill_name_matches(name, NICE_TO_HAVE_SKILLS):
            nice_hits.append(entry)

    # Also scan career descriptions and headline for required skills
    all_text_lower = _normalize(
        headline + " " + summary + " " + " ".join(
            j.get("description", "") + " " + j.get("title", "")
            for j in career
        )
    )
    text_required_hits = [t for t in REQUIRED_SKILLS if t in all_text_lower]
    text_nice_hits = [t for t in NICE_TO_HAVE_SKILLS if t in all_text_lower]

    # ------------------------------------------------------------------
    # Career analysis
    # ------------------------------------------------------------------
    total_career_months = 0
    company_types: list[str] = []  # PRODUCT / CONSULTING / UNKNOWN
    ai_role_months = 0
    tenures: list[int] = []
    has_production_keywords = False
    all_companies: list[str] = []
    role_descriptions: list[str] = []
    industries: list[str] = []

    production_kw = [
        "deployed", "production", "shipped", "real users", "at scale",
        "latency", "throughput", "serving", "inference", "ranking system",
        "search system", "recommendation", "retrieval", "embedding",
        "vector", "similarity", "ranking", "retrieval system",
    ]

    for job in career:
        dur = int(job.get("duration_months", 0) or 0)
        total_career_months += dur
        tenures.append(dur)

        comp = job.get("company", "")
        all_companies.append(comp)
        industries.append(job.get("industry", ""))
        desc = job.get("description", "")
        title_j = job.get("title", "")
        role_descriptions.append(desc + " " + title_j)

        if _is_consulting(comp):
            company_types.append("CONSULTING")
        else:
            company_types.append("PRODUCT")

        if _contains_any(desc + " " + title_j, production_kw):
            has_production_keywords = True

        if _is_ai_title(title_j):
            ai_role_months += dur

    consulting_only = bool(company_types) and all(ct == "CONSULTING" for ct in company_types)
    has_product_experience = "PRODUCT" in company_types

    avg_tenure = (sum(tenures) / len(tenures)) if tenures else 0
    title_chaser = avg_tenure < 18 and len(tenures) > 2

    # ------------------------------------------------------------------
    # Location / availability
    # ------------------------------------------------------------------
    india_locations = {"india", "pune", "noida", "bangalore", "bengaluru",
                       "mumbai", "hyderabad", "delhi", "gurgaon", "gurugram",
                       "chennai", "kolkata", "ahmedabad", "ncr"}
    in_india = _normalize(country) == "india" or any(
        loc in _normalize(location) for loc in india_locations
    )
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    # ------------------------------------------------------------------
    # Education
    # ------------------------------------------------------------------
    best_tier = "unknown"
    for edu in education:
        t = edu.get("tier", "unknown")
        if TIER_SCORES.get(t, 0) > TIER_SCORES.get(best_tier, 0):
            best_tier = t

    # ------------------------------------------------------------------
    # Redrob signals
    # ------------------------------------------------------------------
    open_to_work = bool(signals.get("open_to_work_flag", False))
    notice_days = int(signals.get("notice_period_days", 90) or 90)
    response_rate = float(signals.get("recruiter_response_rate", 0) or 0)
    interview_rate = float(signals.get("interview_completion_rate", 0) or 0)
    profile_complete = float(signals.get("profile_completeness_score", 0) or 0)
    github_score = float(signals.get("github_activity_score", -1))
    last_active_str = signals.get("last_active_date")
    last_active = _parse_date(last_active_str)
    days_since_active = (_today() - last_active).days if last_active else 999
    work_mode = signals.get("preferred_work_mode", "flexible")
    saved_30d = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    offer_rate = float(signals.get("offer_acceptance_rate", -1))
    verified = bool(signals.get("verified_email", False)) and bool(signals.get("verified_phone", False))

    # ------------------------------------------------------------------
    # Build full text for embedding
    # ------------------------------------------------------------------
    skill_text = " ".join(skill_names)
    cert_text = " ".join(c.get("name", "") for c in certs)
    career_text = " ".join(
        f"{j.get('title','')} {j.get('company','')} {j.get('description','')}"
        for j in career
    )
    full_text = (
        f"{current_title} {headline} {summary} "
        f"{skill_text} {cert_text} {career_text}"
    ).strip()

    return {
        "candidate_id": cid,
        "yoe": yoe,
        "current_title": current_title,
        "current_company": current_company,
        "country": country,
        "location": location,
        "in_india": in_india,
        "willing_to_relocate": willing_to_relocate,
        # Skills
        "required_hits": required_hits,
        "nice_hits": nice_hits,
        "text_required_hits": text_required_hits,
        "text_nice_hits": text_nice_hits,
        "skill_names": skill_names,
        "num_skills": len(skills),
        # Career
        "total_career_months": total_career_months,
        "ai_role_months": ai_role_months,
        "consulting_only": consulting_only,
        "has_product_experience": has_product_experience,
        "company_types": company_types,
        "all_companies": all_companies,
        "avg_tenure_months": avg_tenure,
        "title_chaser": title_chaser,
        "has_production_keywords": has_production_keywords,
        "industries": industries,
        "role_descriptions": role_descriptions,
        "is_ai_current_title": _is_ai_title(current_title),
        "is_non_ai_current_title": _is_non_ai_title(current_title),
        # Education
        "best_edu_tier": best_tier,
        # Availability signals
        "open_to_work": open_to_work,
        "notice_days": notice_days,
        "response_rate": response_rate,
        "interview_rate": interview_rate,
        "profile_complete": profile_complete,
        "github_score": github_score,
        "days_since_active": days_since_active,
        "work_mode": work_mode,
        "saved_30d": saved_30d,
        "offer_rate": offer_rate,
        "verified": verified,
        # Text for embedding
        "full_text": full_text,
    }
