"""
CSV loader utilities — loads all dataset CSVs and returns typed dicts.
"""

import csv
import os


def load_claims(filepath: str) -> list:
    """Load claims.csv or sample_claims.csv and return list of dicts."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            cleaned = {k.strip(): v.strip() if v else v for k, v in row.items()}
            rows.append(cleaned)
    return rows


def load_user_history(filepath: str) -> dict:
    """Load user_history.csv and return dict keyed by user_id."""
    history = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() if v else v for k, v in row.items()}
            uid = cleaned.get("user_id", "")
            history[uid] = {
                "user_id": uid,
                "past_claim_count": int(cleaned.get("past_claim_count", 0)),
                "accept_claim": int(cleaned.get("accept_claim", 0)),
                "manual_review_claim": int(cleaned.get("manual_review_claim", 0)),
                "rejected_claim": int(cleaned.get("rejected_claim", 0)),
                "last_90_days_claim_count": int(cleaned.get("last_90_days_claim_count", 0)),
                "history_flags": cleaned.get("history_flags", "none"),
                "history_summary": cleaned.get("history_summary", ""),
            }
    return history


def load_evidence_requirements(filepath: str) -> list:
    """Load evidence_requirements.csv and return list of dicts."""
    reqs = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() if v else v for k, v in row.items()}
            reqs.append(cleaned)
    return reqs


def get_relevant_requirements(requirements: list, claim_object: str) -> list:
    """Filter evidence requirements relevant to a given claim_object."""
    relevant = []
    for req in requirements:
        obj = req.get("claim_object", "").lower()
        if obj == "all" or obj == claim_object.lower():
            relevant.append(req)
    return relevant


def get_default_user_history() -> dict:
    """Return safe defaults for a user not found in user_history.csv."""
    return {
        "user_id": "unknown",
        "past_claim_count": 0,
        "accept_claim": 0,
        "manual_review_claim": 0,
        "rejected_claim": 0,
        "last_90_days_claim_count": 0,
        "history_flags": "none",
        "history_summary": "No prior history available",
    }
