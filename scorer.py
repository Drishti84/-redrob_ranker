"""
Weighted multi-signal scoring engine.
All functions are pure — takes pre-extracted feature dicts, returns 0..1 floats.

Final score = 0.35*skill + 0.25*career + 0.20*availability + 0.15*semantic + 0.05*education
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "skill":        0.35,
    "career":       0.25,
    "availability": 0.20,
    "semantic":     0.15,
    "education":    0.05,
}

TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.45,
    "tier_4": 0.20,
    "unknown": 0.30,
}

# Consulting firms — penalize if entire career is services
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra",
    "mphasis", "hexaware", "l&t infotech", "ltimindtree", "mindtree",
}

# ---------------------------------------------------------------------------
# Component: skill_match_score
# ---------------------------------------------------------------------------

def skill_match_score(feat: dict[str, Any]) -> float:
    """
    Score based on how well skills + career text match JD requirements.
    Required skills coverage is the primary signal; nice-to-have adds bonus.
    Applies a penalty when the current title is completely off-domain — catches
    keyword stuffers (e.g., Marketing Manager who lists every AI skill).
    """
    required_hits = feat.get("required_hits", [])
    nice_hits = feat.get("nice_hits", [])
    text_required = feat.get("text_required_hits", [])
    text_nice = feat.get("text_nice_hits", [])
    is_non_ai_title = feat.get("is_non_ai_current_title", False)
    ai_months = feat.get("ai_role_months", 0)

    # --- Required skills from explicit skill list ---
    if required_hits:
        avg_req = sum(h["weight"] for h in required_hits) / len(required_hits)
        coverage = min(1.0, len(required_hits) / 6.0)
        req_score = avg_req * coverage
    else:
        req_score = 0.0

    # --- Boost from career text mentions (catches implied skills not listed explicitly) ---
    text_boost = min(0.25, len(text_required) * 0.03)

    # --- Nice-to-have bonus ---
    nice_score = min(0.15, len(nice_hits) * 0.03 + len(text_nice) * 0.015)

    raw = req_score + text_boost + nice_score

    # --- Keyword stuffer penalty ---
    # If title is off-domain AND they have very little AI work history,
    # their listed AI skills are likely aspirational, not demonstrated.
    if is_non_ai_title and ai_months < 12:
        raw *= 0.40  # heavy discount — skills without career evidence are suspect

    return min(1.0, raw)


# ---------------------------------------------------------------------------
# Component: career_quality_score
# ---------------------------------------------------------------------------

def career_quality_score(feat: dict[str, Any]) -> float:
    """
    Evaluates career trajectory for fit with the JD profile:
    - Product company AI/ML experience preferred
    - Consulting-only = heavy penalty
    - Title mismatch (Accountant, Mechanic, etc.) = heavy penalty
    - Title chaser (avg tenure < 18m) = penalty
    - Production deployment keywords = bonus
    """
    score = 0.5  # neutral start

    yoe = feat.get("yoe", 0)
    ai_months = feat.get("ai_role_months", 0)
    consulting_only = feat.get("consulting_only", False)
    has_product = feat.get("has_product_experience", False)
    avg_tenure = feat.get("avg_tenure_months", 0)
    title_chaser = feat.get("title_chaser", False)
    has_production = feat.get("has_production_keywords", False)
    is_ai_title = feat.get("is_ai_current_title", False)
    is_non_ai_title = feat.get("is_non_ai_current_title", False)
    current_company = (feat.get("current_company", "") or "").lower()

    # --- Current title alignment ---
    if is_ai_title:
        score += 0.20
    elif is_non_ai_title:
        score -= 0.35  # Strong penalty — Marketing Manager etc.

    # --- AI/ML experience by months ---
    ai_years = ai_months / 12.0
    if ai_years >= 4:
        score += 0.20
    elif ai_years >= 2:
        score += 0.10
    elif ai_years >= 1:
        score += 0.04

    # --- Experience band: JD wants 5-9 years ---
    if 5 <= yoe <= 9:
        score += 0.10
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        score += 0.05
    elif yoe < 3:
        score -= 0.10

    # --- Consulting-only penalty ---
    if consulting_only:
        score -= 0.30
    elif not has_product:
        score -= 0.10

    # --- Production deployment evidence ---
    if has_production:
        score += 0.12

    # --- Title chaser penalty ---
    if title_chaser:
        score -= 0.15

    # --- Currently at consulting but has product experience is OK ---
    is_currently_consulting = any(
        firm in current_company for firm in CONSULTING_FIRMS
    )
    if is_currently_consulting and has_product:
        score += 0.0  # neutral — they have product background, just currently at services

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Component: availability_score
# ---------------------------------------------------------------------------

def availability_score(feat: dict[str, Any]) -> float:
    """
    Behavioral signals: is this person actually hirable right now?
    The JD explicitly says to down-weight unavailable candidates.
    """
    score = 0.0

    open_to_work = feat.get("open_to_work", False)
    days_active = feat.get("days_since_active", 999)
    response_rate = feat.get("response_rate", 0.0)
    interview_rate = feat.get("interview_rate", 0.0)
    notice_days = feat.get("notice_days", 90)
    relocate = feat.get("willing_to_relocate", False)
    in_india = feat.get("in_india", False)
    profile_complete = feat.get("profile_complete", 0.0)
    verified = feat.get("verified", False)
    saved_30d = feat.get("saved_30d", 0)
    work_mode = feat.get("work_mode", "flexible")
    offer_rate = feat.get("offer_rate", -1.0)
    github_score = feat.get("github_score", -1.0)

    # Open to work: most important binary signal
    if open_to_work:
        score += 0.20

    # Recency — last active within 30 days = great, 6+ months = cold
    if days_active <= 7:
        score += 0.20
    elif days_active <= 30:
        score += 0.16
    elif days_active <= 90:
        score += 0.10
    elif days_active <= 180:
        score += 0.04
    else:
        score += 0.0  # cold lead

    # Recruiter response rate
    score += 0.18 * response_rate

    # Interview completion rate
    score += 0.12 * interview_rate

    # Notice period — JD wants sub-30d, can buy out up to 30
    if notice_days <= 30:
        score += 0.10
    elif notice_days <= 60:
        score += 0.07
    elif notice_days <= 90:
        score += 0.04
    else:
        score += 0.0

    # Location fit: India + Pune/Noida preferred (hybrid role)
    if in_india:
        score += 0.07
    elif relocate:
        score += 0.03

    # Profile completeness
    score += 0.05 * (profile_complete / 100.0)

    # Verification
    if verified:
        score += 0.03

    # Recruiter interest signal: saved by recruiters recently
    if saved_30d >= 3:
        score += 0.03
    elif saved_30d >= 1:
        score += 0.01

    # Offer acceptance — if they've been offered but keep declining, caution
    if offer_rate >= 0.5:
        score += 0.01
    elif 0 <= offer_rate < 0.3:
        score -= 0.02

    # GitHub activity — positive for an AI engineer
    if github_score >= 40:
        score += 0.02
    elif github_score >= 10:
        score += 0.01

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Component: education_score
# ---------------------------------------------------------------------------

def education_score(feat: dict[str, Any]) -> float:
    return TIER_SCORES.get(feat.get("best_edu_tier", "unknown"), 0.30)


# ---------------------------------------------------------------------------
# Final composite score
# ---------------------------------------------------------------------------

def composite_score(
    feat: dict[str, Any],
    semantic_sim: float = 0.5,
) -> dict[str, float]:
    """
    Compute all component scores and the weighted composite.
    semantic_sim: cosine similarity from FAISS/embeddings, passed in from rank.py.
    Returns dict with all component scores + 'final'.
    """
    sk = skill_match_score(feat)
    ca = career_quality_score(feat)
    av = availability_score(feat)
    ed = education_score(feat)
    se = max(0.0, min(1.0, semantic_sim))

    final = (
        WEIGHTS["skill"]        * sk
        + WEIGHTS["career"]     * ca
        + WEIGHTS["availability"] * av
        + WEIGHTS["semantic"]   * se
        + WEIGHTS["education"]  * ed
    )

    return {
        "skill":        round(sk, 4),
        "career":       round(ca, 4),
        "availability": round(av, 4),
        "semantic":     round(se, 4),
        "education":    round(ed, 4),
        "final":        round(final, 4),
    }
