"""
Per-candidate reasoning string generator.
Produces specific, fact-grounded 1-2 sentence reasonings.
No templates — branches on what makes each candidate distinctive.
Stage 4 reviewers check for: specific facts, JD connection, honest concerns, no hallucination, variation.
"""

from __future__ import annotations

from typing import Any


def _fmt_yoe(yoe: float) -> str:
    return f"{yoe:.1f} yrs"


def _top_skills(feat: dict) -> str:
    hits = feat.get("required_hits", [])
    if not hits:
        hits = feat.get("nice_hits", [])
    top = sorted(hits, key=lambda h: h["weight"], reverse=True)[:3]
    names = [h["name"] for h in top]
    if not names:
        names = feat.get("skill_names", [])[:2]
    return ", ".join(names) if names else "general software skills"


def _title_company(feat: dict) -> str:
    title = feat.get("current_title", "")
    company = feat.get("current_company", "")
    if title and company:
        return f"{title} at {company}"
    return title or company or "unknown role"


def generate_reasoning(
    feat: dict[str, Any],
    scores: dict[str, float],
    rank: int,
) -> str:
    """
    Returns a 1-2 sentence reasoning string for one candidate.
    Pulls specific facts from feat; tone calibrated to rank.
    """
    yoe = feat.get("yoe", 0)
    title_co = _title_company(feat)
    top_sk = _top_skills(feat)
    open_to_work = feat.get("open_to_work", False)
    notice = feat.get("notice_days", 90)
    days_active = feat.get("days_since_active", 999)
    response_rate = feat.get("response_rate", 0.0)
    consulting_only = feat.get("consulting_only", False)
    in_india = feat.get("in_india", False)
    relocate = feat.get("willing_to_relocate", False)
    ai_months = feat.get("ai_role_months", 0)
    has_production = feat.get("has_production_keywords", False)
    is_ai_title = feat.get("is_ai_current_title", False)
    is_non_ai_title = feat.get("is_non_ai_current_title", False)
    edu_tier = feat.get("best_edu_tier", "unknown")
    github = feat.get("github_score", -1)
    skill_score = scores.get("skill", 0)
    career_score = scores.get("career", 0)
    avail_score = scores.get("availability", 0)
    final = scores.get("final", 0)

    parts: list[str] = []

    # ---------------------------------------------------------------
    # Sentence 1: lead with the strongest positive signal
    # ---------------------------------------------------------------
    if rank <= 10:
        # Top tier — lead with what makes them exceptional
        if is_ai_title and ai_months >= 36:
            parts.append(
                f"{title_co} with {_fmt_yoe(yoe)} experience; "
                f"{ai_months // 12}+ yrs in AI/ML roles including {top_sk}"
            )
        elif skill_score >= 0.6:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}); strong required-skill coverage "
                f"including {top_sk}"
            )
        elif has_production:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}) with demonstrated production deployment "
                f"experience; matches core JD profile"
            )
        else:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}); {top_sk} — "
                f"profile aligns with Senior AI Engineer requirements"
            )

    elif rank <= 30:
        if skill_score >= 0.4:
            parts.append(
                f"{title_co} with {_fmt_yoe(yoe)} and skills in {top_sk}"
            )
        elif career_score >= 0.5:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}) with solid AI/ML career history; {top_sk}"
            )
        else:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}); partial JD skill match ({top_sk})"
            )

    else:
        # Rank 31-100: be more neutral/honest
        if is_non_ai_title:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}); adjacent background — "
                f"some JD-relevant skills ({top_sk}) but title is off-domain"
            )
        elif consulting_only:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}); skills include {top_sk} but "
                f"career history is exclusively services/consulting firms"
            )
        else:
            parts.append(
                f"{title_co} ({_fmt_yoe(yoe)}); relevant skills: {top_sk}"
            )

    # ---------------------------------------------------------------
    # Sentence 2: availability + key concern or strong positive
    # ---------------------------------------------------------------
    avail_parts: list[str] = []
    concern_parts: list[str] = []

    # Availability positives
    if open_to_work and notice <= 30:
        avail_parts.append(f"open to work, {notice}d notice")
    elif open_to_work:
        avail_parts.append(f"open to work ({notice}d notice)")
    elif notice <= 30:
        avail_parts.append(f"{notice}d notice (may not be actively looking)")

    if days_active <= 14:
        avail_parts.append("active on platform recently")
    elif days_active > 120:
        concern_parts.append(f"last active {days_active}d ago")

    if response_rate >= 0.7:
        avail_parts.append(f"high recruiter response rate ({response_rate:.0%})")
    elif response_rate < 0.2 and rank <= 50:
        concern_parts.append(f"low recruiter response rate ({response_rate:.0%})")

    # Location
    if in_india:
        avail_parts.append("India-based")
    elif relocate:
        avail_parts.append("willing to relocate")
    else:
        concern_parts.append("outside India, not willing to relocate")

    # GitHub bonus for top candidates
    if github >= 40 and rank <= 20:
        avail_parts.append(f"active GitHub (score {github:.0f})")

    # Education signal for top candidates
    if edu_tier in ("tier_1", "tier_2") and rank <= 30:
        avail_parts.append(f"{edu_tier.replace('_', ' ')} institution")

    # Concerns for honesty
    if is_non_ai_title and rank <= 50:
        concern_parts.append("current role is not AI/ML-aligned")
    if consulting_only and rank <= 40:
        concern_parts.append("consulting-only background")
    if yoe < 4 and rank <= 30:
        concern_parts.append(f"only {_fmt_yoe(yoe)} experience (JD wants 5-9)")
    if yoe > 12 and rank <= 30:
        concern_parts.append("above experience band (>12 yrs)")

    # Build sentence 2
    sentence2_parts: list[str] = []
    if avail_parts:
        sentence2_parts.append(", ".join(avail_parts))
    if concern_parts and rank > 20:
        sentence2_parts.append("concern: " + "; ".join(concern_parts))
    elif concern_parts and rank <= 20:
        sentence2_parts.append("note: " + "; ".join(concern_parts))

    if sentence2_parts:
        parts.append("; ".join(sentence2_parts))

    return ". ".join(parts).rstrip(".") + "."
