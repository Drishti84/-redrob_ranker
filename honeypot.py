"""
Honeypot detection — identifies candidates with impossible profiles.
The dataset contains ~80 synthetic traps; ranking them hurts NDCG and
triggers disqualification if > 10% of top-100 are honeypots.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _today() -> date:
    return date.today()


def is_honeypot(candidate: dict[str, Any]) -> tuple[bool, str]:
    """
    Returns (is_honeypot, reason).
    Checks for impossible profile patterns.
    """
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0) or 0)

    # ---------------------------------------------------------------
    # Check 1: career date math vs claimed duration_months
    # ---------------------------------------------------------------
    for job in career:
        start = _parse_date(job.get("start_date"))
        end_raw = job.get("end_date")
        end = _parse_date(end_raw) if end_raw else _today()
        claimed = int(job.get("duration_months", 0) or 0)

        if start and end:
            actual_months = (end - start).days / 30.44
            diff = abs(actual_months - claimed)
            # Allow 2-month slop for rounding; flag if > 12 months off
            if diff > 12:
                return (
                    True,
                    f"duration mismatch at {job.get('company','?')}: "
                    f"claimed {claimed}m vs date-derived {actual_months:.0f}m",
                )
            # End date before start date
            if end < start:
                return (True, f"end_date before start_date at {job.get('company','?')}")

    # ---------------------------------------------------------------
    # Check 2: expert skill with 0 months used
    # ---------------------------------------------------------------
    for sk in skills:
        if sk.get("proficiency") == "expert" and (sk.get("duration_months") or 0) == 0:
            return (
                True,
                f"expert proficiency in '{sk.get('name','?')}' with 0 months used",
            )

    # ---------------------------------------------------------------
    # Check 3: claimed YoE far exceeds sum of career history
    # ---------------------------------------------------------------
    total_career_months = sum(int(j.get("duration_months", 0) or 0) for j in career)
    if career and yoe > 0:
        derived_yoe = total_career_months / 12.0
        # If claimed YoE is 5+ years more than what the career history shows
        if yoe - derived_yoe > 5:
            return (
                True,
                f"claimed YoE {yoe:.1f} but career history totals only {derived_yoe:.1f} yrs",
            )

    # ---------------------------------------------------------------
    # Check 4: Multiple overlapping concurrent jobs (impossible timeline)
    # ---------------------------------------------------------------
    active_jobs = []
    for job in career:
        start = _parse_date(job.get("start_date"))
        end_raw = job.get("end_date")
        end = _parse_date(end_raw) if end_raw else _today()
        if start:
            active_jobs.append((start, end))

    # Sort by start date and check for overlaps > 6 months
    active_jobs.sort(key=lambda x: x[0])
    for i in range(len(active_jobs) - 1):
        s1, e1 = active_jobs[i]
        s2, e2 = active_jobs[i + 1]
        if s2 < e1:  # overlap
            overlap_days = (min(e1, e2) - s2).days
            if overlap_days > 180:  # more than 6 months overlap = suspicious
                return (True, f"jobs overlap by {overlap_days} days")

    # ---------------------------------------------------------------
    # Check 5: Future start dates
    # ---------------------------------------------------------------
    today = _today()
    for job in career:
        start = _parse_date(job.get("start_date"))
        if start and start > today:
            return (True, f"future start_date {start} at {job.get('company','?')}")

    # ---------------------------------------------------------------
    # Check 6: Absurd skill count with all expert + 0 duration
    # ---------------------------------------------------------------
    expert_zero_dur = sum(
        1 for sk in skills
        if sk.get("proficiency") in ("expert", "advanced")
        and (sk.get("duration_months") or 0) < 3
    )
    if expert_zero_dur >= 5:
        return (True, f"{expert_zero_dur} expert/advanced skills with < 3 months use")

    return (False, "")


def filter_honeypots(
    features_list: list[dict[str, Any]],
    candidates_map: dict[str, dict[str, Any]],
) -> tuple[list[dict], list[dict]]:
    """
    Split features_list into (clean, honeypots).
    candidates_map: {candidate_id: raw candidate dict}
    """
    clean, traps = [], []
    for feat in features_list:
        cid = feat["candidate_id"]
        raw = candidates_map.get(cid, {})
        trap, _ = is_honeypot(raw)
        if trap:
            traps.append(feat)
        else:
            clean.append(feat)
    return clean, traps
