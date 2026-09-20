"""OpenRouter Jev adapter for bounded Calyber pricing and matching decisions."""

import json
import urllib.error
import urllib.request


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if key != "rider_ref"
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _price_key(price):
    return f"price_{int(round(price * 100)):02d}"


def _price_description(price):
    if price < 0.60:
        posture = "an unusually aggressive conversion-first quote"
    elif price < 0.75:
        posture = "a balanced quote that trades some margin for conversion"
    elif price < 0.90:
        posture = "a margin-first quote requiring reasonable willingness to pay"
    else:
        posture = "a premium quote with substantial non-conversion risk"
    return (
        f"Quote ${price:.2f} per mile. This is {posture}. Select it only when it "
        "offers the best expected profit per arrival for the supplied state."
    )


def build_jev_request(model, summary, kpis, recent, price_grid, advisor_text=""):
    """Build one Decisions request with independent price and match choices."""
    candidates = summary.get("candidates") or []
    state = {
        "objective": (
            "Maximize long-run profit per minute while maintaining enough conversion "
            "and matching throughput. Prices are dollars per rider-mile."
        ),
        "operating_cost_per_mile": 0.70,
        "current_state": _json_safe(summary),
        "rolling_kpis": _json_safe(kpis or {}),
        "recent_outcomes": _json_safe(recent or []),
        "strategic_advisor_guidance": advisor_text or "No advisor guidance is available yet.",
        "constraints": [
            "Choose exactly one listed price.",
            "Choose WAIT when no feasible match exists.",
            "Prefer matches with meaningful route savings, especially as queues age.",
            "Do not maximize conversion at the expense of negative expected margin.",
        ],
    }

    price_criteria = {
        _price_key(price): _price_description(price) for price in price_grid
    }
    match_criteria = {
        "wait": (
            "Keep the incoming rider waiting. Select when there is no candidate or "
            "when every available match is too weak relative to the value of waiting."
        )
    }
    for candidate in candidates:
        rank = int(candidate["rank"])
        match_criteria[f"match_{rank}"] = (
            f"Immediately match candidate {rank}: savings rate "
            f"{candidate['savings_rate']:.1%}, matched cost estimate "
            f"${candidate['cost_if_matched']:.3f}/mile, waiting age "
            f"{candidate['waiting_age_sec']:.1f} seconds."
        )

    return {
        "model": model,
        "state": state,
        "questions": {
            "price": {
                "type": "choice",
                "instructions": (
                    "Which allowed quote maximizes expected profit for this arrival, "
                    "considering conversion risk, operating cost, match savings, "
                    "recent outcomes, and the strategic advisor guidance?"
                ),
                "criteria": price_criteria,
            },
            "match": {
                "type": "choice",
                "instructions": (
                    "Should the incoming rider wait or be matched immediately, and "
                    "if matched, which candidate gives the best operational outcome?"
                ),
                "criteria": match_criteria,
            },
        },
    }


def parse_jev_response(response_data, price_grid, num_candidates):
    """Convert typed Jev answers into the decision dictionary used by the agent."""
    answers = response_data.get("answers") or {}
    price_answer = answers.get("price") or {}
    match_answer = answers.get("match") or {}

    price_choice = price_answer.get("choice")
    match_choice = match_answer.get("choice")
    if not isinstance(price_choice, str) or not price_choice.startswith("price_"):
        raise ValueError("Jev response is missing a valid price choice")
    if not isinstance(match_choice, str):
        raise ValueError("Jev response is missing a valid match choice")

    cents = int(price_choice.split("_", 1)[1])
    price = cents / 100.0
    if price not in price_grid:
        raise ValueError(f"Jev selected a price outside the grid: {price}")

    candidate_rank = None
    action = "WAIT"
    if match_choice != "wait":
        if not match_choice.startswith("match_"):
            raise ValueError(f"Jev selected an unknown match action: {match_choice}")
        candidate_rank = int(match_choice.split("_", 1)[1])
        if not 1 <= candidate_rank <= num_candidates:
            raise ValueError(f"Jev selected an invalid candidate: {candidate_rank}")
        action = "MATCH"

    price_conf = price_answer.get("confidence")
    match_conf = match_answer.get("confidence")
    confidence_text = []
    if isinstance(price_conf, (int, float)):
        confidence_text.append(f"price confidence {price_conf:.2f}")
    if isinstance(match_conf, (int, float)):
        confidence_text.append(f"match confidence {match_conf:.2f}")
    confidence_suffix = f" ({', '.join(confidence_text)})" if confidence_text else ""

    return {
        "price": price,
        "decision": action,
        "candidate_rank": candidate_rank,
        "reasoning": f"Jev selected {price_choice} and {match_choice}{confidence_suffix}.",
        "raw": json.dumps(response_data, separators=(",", ":")),
        "jev_answers": answers,
    }


def query_jev(
    api_key,
    endpoint,
    model,
    summary,
    kpis,
    recent,
    price_grid,
    advisor_text="",
    timeout=10.0,
):
    """Call OpenRouter's Decisions endpoint and return a Calyber decision."""
    payload = build_jev_request(
        model, summary, kpis, recent, price_grid, advisor_text=advisor_text
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/calyber",
            "X-OpenRouter-Title": "Calyber AI Agent",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter Decisions HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter Decisions request failed: {exc.reason}") from exc

    return parse_jev_response(
        response_data, price_grid, len(summary.get("candidates") or [])
    )
