"""
Parser / Validator — converts the LLM's JSON response into a validated
decision dict.  Now also extracts the "reasoning" field (chain-of-thought)
that the agent produces before each decision.
"""

import json
import re

from .config import PRICE_GRID, FALLBACK_PRICE, MIN_SAVINGS_THRESHOLD


def _extract_json(text):
    """Pull the first JSON object out of *text*, tolerating markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _snap_to_grid(value, grid):
    """Return the grid value closest to *value*."""
    return min(grid, key=lambda g: abs(g - value))


def parse_response(text, price_grid=PRICE_GRID, num_candidates=0):
    """
    Parse and validate an LLM response string.

    Returns a decision dict on success::

        {"price": float, "decision": "WAIT"|"MATCH",
         "candidate_rank": int|None, "reasoning": str, "raw": str}

    Returns None if the response cannot be parsed into a valid decision.
    """
    if not text:
        return None

    raw_json = _extract_json(text)
    if raw_json is None:
        return None

    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    # ── extract reasoning (chain-of-thought) ────────────────────────
    reasoning = obj.get("reasoning", "")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    else:
        reasoning = ""

    # ── price ───────────────────────────────────────────────────────
    price_raw = obj.get("price_choice")
    if price_raw is None:
        return None
    try:
        price_val = float(price_raw)
    except (TypeError, ValueError):
        return None

    price_val = _snap_to_grid(price_val, price_grid)

    # ── decision ────────────────────────────────────────────────────
    decision_raw = obj.get("decision", "").upper().strip()

    # Handle LLM writing "MATCH 1" or "MATCH candidate 2" in the decision field
    candidate_rank = None
    embedded_rank = re.search(r"MATCH\s*(?:CANDIDATE\s*)?(\d+)", decision_raw)
    if embedded_rank:
        decision_raw = "MATCH"
        candidate_rank = int(embedded_rank.group(1))

    if decision_raw not in ("WAIT", "MATCH"):
        return None

    # ── candidate rank ──────────────────────────────────────────────
    if decision_raw == "MATCH" and candidate_rank is None:
        rank_raw = obj.get("candidate_rank")
        if rank_raw is None:
            return None
        try:
            candidate_rank = int(rank_raw)
        except (TypeError, ValueError):
            return None

    if decision_raw == "MATCH":
        if candidate_rank is None:
            return None
        if candidate_rank < 1 or candidate_rank > num_candidates:
            if num_candidates == 0:
                decision_raw = "WAIT"
                candidate_rank = None
            else:
                candidate_rank = min(max(candidate_rank, 1), num_candidates)

    return {
        "price": price_val,
        "decision": decision_raw,
        "candidate_rank": candidate_rank,
        "reasoning": reasoning,
        "raw": text,
    }


def fallback_decision(summary, default_price=FALLBACK_PRICE,
                      min_savings=MIN_SAVINGS_THRESHOLD, price_grid=PRICE_GRID):
    """
    Deterministic backup: mid-range price + greedy best-savings match.
    Used when the LLM times out, returns garbage, or is unavailable.
    """
    price = _snap_to_grid(default_price, price_grid)

    candidates = summary.get("candidates", [])
    best = None
    for c in candidates:
        if c["savings_rate"] >= min_savings:
            best = c
            break

    if best is not None:
        return {
            "price": price,
            "decision": "MATCH",
            "candidate_rank": best["rank"],
            "reasoning": "fallback: greedy best-savings",
            "raw": "__fallback__",
        }

    return {
        "price": price,
        "decision": "WAIT",
        "candidate_rank": None,
        "reasoning": "fallback: no good candidates",
        "raw": "__fallback__",
    }
