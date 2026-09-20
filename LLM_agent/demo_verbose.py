#!/usr/bin/env python3
"""
Verbose Demo Runner — runs the LLM agent on synthetic riders with a
live evolving queue, prints every stage of the pipeline, and saves
results to CSV files.  Designed for real-time demo.

Usage:
    export OPENAI_API_KEY="sk-proj-..."
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python -m LLM_agent.demo_verbose              # default 10 rounds
    python -m LLM_agent.demo_verbose --rounds 15  # more rounds

Coach (advisor) defaults to gpt-5.4 and refreshes every 25 arrivals;
the decision model defaults to Jev 1.13 through OpenRouter. Override e.g.:
    export CALYBER_DECISION_BACKEND="jev"
    export CALYBER_JEV_MODEL="typesafe/jev-1.13"
    export CALYBER_ADVISOR_INTERVAL="25"
    export CALYBER_ADVISOR_MODEL="gpt-5.4"
    # Legacy alias: CALYBER_ORCHESTRATOR_MODEL. Disable coach: CALYBER_ADVISOR_MODEL=

Output CSVs (saved to LLM_agent/results/):
    stage1_perception.csv     — state summary per round
    stage2_memory.csv         — agent KPIs per round
    stage3_candidates.csv     — all candidate matches per round
    stage_coach_advisor.csv   — coach advice text + JSON sent to advisor (orchestrator input)
    stage4_llm_call.csv       — decision LLM metadata per round
    stage5_decisions.csv      — parsed decisions per round
    stage6_reasoning.csv      — chain-of-thought per round
    master_summary.csv        — one-row-per-round with key fields

Set CALYBER_DECISION_BACKEND=openai and CALYBER_LLM_MODEL to restore the old
chat decision maker. Disable the coach with CALYBER_ADVISOR_MODEL=.
"""

import sys, os, time, csv, textwrap, argparse
import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rider import rider as Rider
from LLM_agent.config import PRICE_GRID, COST_PER_MILE, MAX_CANDIDATES
from LLM_agent.state_summary import summarize_state
from LLM_agent.prompt_builder import SYSTEM_PROMPT
from LLM_agent.agent_core import CalyberAgent

# ═════════════════════════════════════════════════════════════════════
#  SYNTHETIC RIDER DATA — realistic Chicago ride requests
# ═════════════════════════════════════════════════════════════════════

# (description, arrival_time, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, pickup_area, dropoff_area)
# WTP is drawn from Beta(3,2) at runtime — agent never sees it
SYNTHETIC_ARRIVALS = [
    ("Downtown short (Loop)",        50, 41.8819,-87.6324, 41.9032,-87.6723, 32, 22),
    ("South to North Side",         120, 41.7482,-87.6101, 41.9427,-87.6518, 44,  6),
    ("Midtown reverse",             200, 41.9217,-87.6544, 41.8781,-87.6298,  7, 32),
    ("Near West Side hop",          280, 41.8827,-87.6233, 41.8986,-87.6228,  8,  6),
    ("Far South to Midway",         370, 41.7142,-87.6382, 41.7860,-87.7509, 73, 56),
    ("Lakeview to Loop",            440, 41.9500,-87.6541, 41.8819,-87.6324,  3, 32),
    ("Humboldt to Wicker",          530, 41.9032,-87.6723, 41.8781,-87.6614, 22, 28),
    ("Loop to O'Hare area",         620, 41.8786,-87.6717, 41.9867,-87.6634, 28, 76),
    ("River North to South Loop",   710, 41.8986,-87.6228, 41.8706,-87.6222,  6, 32),
    ("West Town to Pilsen",         800, 41.9049,-87.6499, 41.8559,-87.6618, 24, 31),
    ("Rogers Park to Edgewater",    900, 42.0087,-87.6722, 41.9867,-87.6634,  1, 76),
    ("Bridgeport to Hyde Park",     990, 41.8381,-87.6502, 41.7911,-87.6183, 60, 40),
    ("Wicker Park to Loop",        1080, 41.8843,-87.6328, 41.8819,-87.6324, 28, 32),
    ("Beverly to Lincoln Park",    1170, 41.7220,-87.6680, 41.9217,-87.6544, 72,  7),
    ("Gold Coast to Streeterville", 1260, 41.9066,-87.6290, 41.8919,-87.6129,  8,  8),
]

def generate_wtps(n, seed=42):
    """Draw hidden WTPs from Beta(3,2) — same distribution the real game uses."""
    np.random.seed(seed)
    return beta_dist.rvs(3, 2, size=n)

# Initial queue — 3 riders already waiting when demo starts
INITIAL_QUEUE = [
    Rider(1, 10, 41.8997, -87.6650, 41.9306, -87.6422, 24, 7),    # area 24->7
    Rider(1, 20, 41.8781, -87.6298, 41.9217, -87.6544, 32, 7),    # area 32->7
    Rider(1, 30, 41.9500, -87.6541, 41.8953, -87.6178, 3, 8),     # area 3->8
]

RENEGE_THRESHOLD = 400.0  # riders leave queue after this many seconds

# ── pretty printing ──────────────────────────────────────────────────

GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

DLINE = f"{DIM}{'═' * 72}{RESET}"

def header(text):
    print(f"\n{DLINE}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(DLINE)

def stage(num, title):
    print(f"\n{BOLD}{MAGENTA}  ┌─ STAGE {num}: {title}{RESET}")
    print(f"{MAGENTA}  │{RESET}")

def bullet(label, value, color=RESET):
    print(f"{MAGENTA}  │{RESET}  {DIM}{label}:{RESET} {color}{value}{RESET}")

def sub_bullet(text, color=DIM):
    print(f"{MAGENTA}  │{RESET}    {color}{text}{RESET}")

def stage_end():
    print(f"{MAGENTA}  └{'─' * 50}{RESET}")


def _csv_long_text(s, max_len=20000):
    """CSV-safe long text; truncate very large payloads."""
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n...[truncated]"


# ── CSV collector ────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

class CSVCollector:
    def __init__(self):
        self.tables = {}

    def add(self, table_name, row_dict):
        self.tables.setdefault(table_name, []).append(row_dict)

    def write_all(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for name, rows in self.tables.items():
            if not rows:
                continue
            path = os.path.join(RESULTS_DIR, f"{name}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"  Saved {path}  ({len(rows)} rows)")


# ── main ─────────────────────────────────────────────────────────────

def run_demo(n_rounds):
    agent = CalyberAgent()

    if not agent._client:
        print(f"\n{YELLOW}{BOLD}  WARNING: No OPENAI_API_KEY — using rule-based fallback (no API).{RESET}")
        print("  For Jev decisions: export OPENROUTER_API_KEY=\"sk-or-v1-...\"")
        print("  For strategic advice: export OPENAI_API_KEY=\"sk-proj-...\"")

    n_rounds = min(n_rounds, len(SYNTHETIC_ARRIVALS))
    wtps = generate_wtps(n_rounds)
    queue = list(INITIAL_QUEUE)
    collector = CSVCollector()
    total_time = 0
    total_revenue = 0.0
    total_cost = 0.0
    n_conversions = 0
    n_matches = 0
    n_reneged = 0

    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}   CALYBER LLM AGENT — LIVE DEMO ({n_rounds} rounds){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"\n  Model: {agent._model}  |  Price grid: {PRICE_GRID}")
    print(f"  WTP distribution: Beta(3,2)  |  Agent NEVER sees WTP")
    print(f"  Initial queue: {len(queue)} riders  |  Max candidates: {MAX_CANDIDATES}")
    print(f"  Cost/mile: ${COST_PER_MILE:.2f}  |  Renege threshold: {RENEGE_THRESHOLD}s")

    for i in range(n_rounds):
        desc, arr_time, plat, plon, dlat, dlon, p_area, d_area = SYNTHETIC_ARRIVALS[i]
        hidden_wtp = wtps[i]
        rider = Rider(1, arr_time, plat, plon, dlat, dlon, p_area, d_area)

        # Renege: remove riders who waited too long
        before_q = len(queue)
        queue = [r for r in queue if (arr_time - r.arrival_time) < RENEGE_THRESHOLD]
        reneged_now = before_q - len(queue)
        n_reneged += reneged_now
        total_cost += reneged_now * COST_PER_MILE * 3.0  # approx solo cost for reneged riders

        header(f"ROUND {i+1}/{n_rounds}: {desc}  "
               f"(area {p_area}->{d_area}, solo={rider.solo_length:.2f}mi)")
        if reneged_now:
            print(f"  {RED}  [{reneged_now} rider(s) reneged from queue before this round]{RESET}")

        # ── STAGE 1: Perception ──────────────────────────────────────
        stage(1, "PERCEPTION — State Summarizer")
        summary = summarize_state(queue, rider, max_candidates=MAX_CANDIDATES)
        r_info = summary["incoming_rider"]

        bullet("Incoming", f"area {r_info['pickup_area']}->{r_info['dropoff_area']}, "
                           f"{r_info['solo_length']} mi, t={r_info['arrival_time']}s")
        bullet("Queue", f"{summary['queue_size']} riders "
                        f"(max wait: {summary['max_waiting_age']}s, avg: {summary['avg_waiting_age']}s)")

        collector.add("stage1_perception", {
            "round": i + 1, "description": desc,
            "queue_size": summary["queue_size"],
            "pickup_area": r_info["pickup_area"],
            "dropoff_area": r_info["dropoff_area"],
            "solo_length": r_info["solo_length"],
            "arrival_time": r_info["arrival_time"],
            "max_waiting_age_sec": summary["max_waiting_age"],
            "avg_waiting_age_sec": summary["avg_waiting_age"],
            "num_candidates": len(summary["candidates"]),
            "reneged_this_round": reneged_now,
        })

        if summary["candidates"]:
            bullet("Candidates", f"{len(summary['candidates'])} compatible", GREEN)
            for c in summary["candidates"]:
                sub_bullet(
                    f"#{c['rank']}: savings={c['savings_rate']:.0%}  "
                    f"shared={c['shared_length']}mi  "
                    f"cost=${c['cost_if_matched']:.2f}/mi  "
                    f"waited={c['waiting_age_sec']}s  "
                    f"(area {c['pickup_area']}->{c['dropoff_area']})", GREEN)

                collector.add("stage3_candidates", {
                    "round": i + 1,
                    "candidate_rank": c["rank"],
                    "savings_rate": c["savings_rate"],
                    "shared_length_mi": c["shared_length"],
                    "trip_length_mi": c["trip_length"],
                    "cost_if_matched_per_mi": c["cost_if_matched"],
                    "waiting_age_sec": c["waiting_age_sec"],
                    "candidate_pickup_area": c["pickup_area"],
                    "candidate_dropoff_area": c["dropoff_area"],
                })
        else:
            bullet("Candidates", "0 — no compatible matches", YELLOW)
        stage_end()

        # ── STAGE 2: Memory ──────────────────────────────────────────
        stage(2, "MEMORY — Agent History & KPIs")
        kpis = agent.memory.get_running_kpis()
        recent = agent.memory.get_recent_decisions()

        mem_row = {"round": i + 1}
        if kpis and kpis["window_size"] >= 1:
            bullet("Window", f"{kpis['window_size']} decisions")
            bullet("Avg price", f"${kpis['avg_price_quoted']:.2f}/mi")
            bullet("Conversion", f"{kpis['conversion_rate']:.0%}",
                   GREEN if kpis['conversion_rate'] > 0.5 else RED)
            bullet("Match rate", f"{kpis['match_rate']:.0%}",
                   GREEN if kpis['match_rate'] > 0.4 else YELLOW)
            bullet("Avg savings", f"{kpis['avg_savings_when_matched']:.1%}")
            bullet("Est. profit", f"${kpis['estimated_profit']:.2f}",
                   GREEN if kpis['estimated_profit'] > 0 else RED)
            mem_row.update(kpis)
        else:
            bullet("History", "Cold start — no prior decisions", DIM)
            mem_row.update({
                "window_size": 0, "avg_price_quoted": None,
                "conversion_rate": None, "match_rate": None,
                "avg_savings_when_matched": None,
                "estimated_revenue": None, "estimated_cost": None,
                "estimated_profit": None, "lifetime_arrivals": 0,
                "lifetime_conversions": 0, "lifetime_matches": 0,
            })

        if recent:
            bullet("Recent trail", "")
            for rec in recent[-3:]:
                conv = {True: f"{GREEN}CONV{RESET}",
                        False: f"{RED}REJ{RESET}",
                        None: "?"}[rec.get("converted")]
                sub_bullet(f"R{rec['round']}: ${rec['price']:.2f} | "
                           f"{conv}{RESET} | {rec.get('reasoning','')[:50]}")

        collector.add("stage2_memory", mem_row)
        stage_end()

        # ── STAGE 3: Orchestrator ────────────────────────────────────
        stage(3, "USER PROMPT — Template + optional coach advice")
        user_prompt = agent.build_user_prompt_for_round(summary, kpis, recent, rider)
        pm = dict(agent.last_prompt_meta or {})
        collector.add("stage_coach_advisor", {
            "round": i + 1,
            "compose_source": pm.get("source", ""),
            "decision_model": agent._model,
            "advisor_model": pm.get("advisor_model", ""),
            "advisor_ms": pm.get("advisor_ms") if pm.get("advisor_ms") is not None else "",
            "coach_advice": _csv_long_text(pm.get("coach_advice")),
            "advisor_input_json": _csv_long_text(pm.get("advisor_input_json")),
            "advisor_error": pm.get("advisor_error", ""),
        })
        prompt_lines = user_prompt.split("\n")

        bullet("Prompt", f"{len(user_prompt)} chars, {len(prompt_lines)} lines")
        for section in ["YOUR RUNNING PERFORMANCE", "RECENT DECISIONS",
                        "CURRENT ROUND", "ALLOWED PRICES", "ALLOWED ACTIONS",
                        "COACH ADVICE"]:
            if any(section in l for l in prompt_lines):
                sub_bullet(f"[+] {section}", DIM)

        print(f"{MAGENTA}  │{RESET}")
        for line in prompt_lines:
            print(f"{MAGENTA}  │{RESET}  {DIM}  {line}{RESET}")
        stage_end()

        # ── STAGE 4: LLM Call ────────────────────────────────────────
        stage(4, "LLM DECISION AGENT — API Call")
        t0 = time.time()
        decision = agent.decide(list(queue), rider)
        elapsed_ms = (time.time() - t0) * 1000
        total_time += elapsed_ms

        bullet("Model", agent._model)
        bullet("Latency", f"{elapsed_ms:.0f}ms",
               GREEN if elapsed_ms < 3000 else YELLOW)
        bullet("Source", decision["source"],
               GREEN if decision["source"] == "llm" else RED)

        collector.add("stage4_llm_call", {
            "round": i + 1,
            "model": agent._model,
            "latency_ms": round(elapsed_ms, 1),
            "source": decision["source"],
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "user_prompt_chars": len(user_prompt),
            "raw_response": (decision.get("raw") or "")[:500],
            "compose_source": pm.get("source", ""),
            "advisor_model": pm.get("advisor_model", ""),
            "advisor_ms": pm.get("advisor_ms") if pm.get("advisor_ms") is not None else "",
            "coach_advice_excerpt": ((pm.get("coach_advice") or "")[:500]),
        })
        stage_end()

        # ── STAGE 5: Parser Output ───────────────────────────────────
        stage(5, "PARSER & VALIDATOR — Structured Output")
        price = decision["price"]
        action = decision["decision"]
        cand_rank = decision.get("candidate_rank")

        bullet("Price", f"${price:.2f}/mi", BOLD)
        bullet("Action", action, BOLD)
        if cand_rank:
            bullet("Candidate", f"#{cand_rank}")
        if decision.get("raw"):
            bullet("Raw LLM", decision["raw"][:120], DIM)

        matched_rider_obj = decision.get("matched_rider")
        matched_area = ""
        if matched_rider_obj:
            matched_area = f"{matched_rider_obj.pickup_area}->{matched_rider_obj.dropoff_area}"

        converts = hidden_wtp >= price

        collector.add("stage5_decisions", {
            "round": i + 1, "description": desc,
            "queue_size": summary["queue_size"],
            "price_per_mile": price,
            "hidden_wtp": round(hidden_wtp, 4),
            "converted": int(converts),
            "action": action,
            "candidate_rank": cand_rank,
            "matched_rider_areas": matched_area,
            "source": decision["source"],
            "revenue": round(price * rider.solo_length, 4) if converts else 0,
            "queue_after": None,
        })
        stage_end()

        # ── STAGE 6: Chain-of-Thought ────────────────────────────────
        stage(6, "CHAIN-OF-THOUGHT REASONING")
        reasoning = decision.get("reasoning", "(none)")
        wrapped = textwrap.fill(reasoning, width=60)
        for line in wrapped.split("\n"):
            print(f"{MAGENTA}  │{RESET}  {YELLOW}{BOLD}\"{line}\"{RESET}")

        collector.add("stage6_reasoning", {
            "round": i + 1,
            "reasoning": reasoning,
            "price_chosen": price,
            "hidden_wtp": round(hidden_wtp, 4),
            "converted": int(converts),
            "action_chosen": action,
            "source": decision["source"],
        })
        stage_end()

        # ── OUTCOME ──────────────────────────────────────────────────
        stage("*", "OUTCOME")
        bullet("Hidden WTP", f"${hidden_wtp:.3f}/mi  (agent never sees this)", DIM)
        bullet("Agent price", f"${price:.2f}/mi", BOLD)
        if converts:
            bullet("Conversion", f"YES  (WTP ${hidden_wtp:.3f} >= price ${price:.2f})", GREEN)
        else:
            bullet("Conversion", f"NO   (WTP ${hidden_wtp:.3f} < price ${price:.2f})", RED)

        agent.memory.on_pricing(rider, price, decision,
            {"queue_size": len(queue), "candidates": summary["candidates"]})

        if converts:
            n_conversions += 1
            total_revenue += price * rider.solo_length
            agent.memory.on_matching(rider, decision,
                {"candidates": summary["candidates"]})

            if action == "MATCH" and matched_rider_obj is not None:
                n_matches += 1
                rank = decision.get("candidate_rank")
                cands = summary["candidates"]
                if rank and 1 <= rank <= len(cands):
                    trip_cost = COST_PER_MILE * cands[rank - 1]["trip_length"]
                else:
                    trip_cost = COST_PER_MILE * rider.solo_length
                total_cost += trip_cost
                queue = [w for w in queue if w is not matched_rider_obj]
                bullet("Result", f"CONVERTED + MATCHED with {matched_area} "
                       f"(queue now {len(queue)})", GREEN)
            else:
                total_cost += COST_PER_MILE * rider.solo_length
                queue.append(rider)
                bullet("Result", f"CONVERTED + WAIT (queue now {len(queue)})", CYAN)
        else:
            bullet("Result", "REJECTED price (non-conversion)", RED)

        # update queue_after in the last stage5 row
        collector.tables["stage5_decisions"][-1]["queue_after"] = len(queue)
        agent.pop_cache(rider)

        collector.add("master_summary", {
            "round": i + 1, "description": desc,
            "arrival_time": arr_time,
            "pickup_area": p_area, "dropoff_area": d_area,
            "solo_length": round(rider.solo_length, 4),
            "queue_before": summary["queue_size"],
            "num_candidates": len(summary["candidates"]),
            "price": price, "hidden_wtp": round(hidden_wtp, 4),
            "converted": int(converts),
            "action": action,
            "candidate_rank": cand_rank,
            "matched_areas": matched_area,
            "reasoning": reasoning,
            "source": decision["source"],
            "latency_ms": round(elapsed_ms, 1),
            "queue_after": len(queue),
            "reneged": reneged_now,
            "cumul_revenue": round(total_revenue, 2),
            "cumul_cost": round(total_cost, 2),
            "cumul_profit": round(total_revenue - total_cost, 2),
            "compose_source": pm.get("source", ""),
            "decision_model": agent._model,
            "advisor_model": pm.get("advisor_model", ""),
            "advisor_ms": pm.get("advisor_ms") if pm.get("advisor_ms") is not None else "",
            "coach_advice_excerpt": (pm.get("coach_advice") or "")[:600],
            "advisor_error": pm.get("advisor_error", ""),
        })
        stage_end()

    # ── Write CSVs ───────────────────────────────────────────────────
    header("SAVING RESULTS")
    collector.write_all()

    # ── Final Summary ────────────────────────────────────────────────
    header("GAME SUMMARY")
    stats = agent.stats
    profit = total_revenue - total_cost
    kpis_final = agent.memory.get_running_kpis()

    print(f"""
  {BOLD}Rounds:{RESET}          {n_rounds}
  {BOLD}Total LLM time:{RESET}  {total_time:.0f}ms ({total_time/n_rounds:.0f}ms avg)
  {BOLD}LLM calls:{RESET}       {stats['llm_calls']}
  {BOLD}Advisor calls:{RESET}    {stats.get('advisor_calls', stats.get('orchestrator_calls', 0))}  ({stats.get('advisor_latency_ms', stats.get('orchestrator_latency_ms', 0)):.0f}ms)
  {BOLD}Parse failures:{RESET}  {stats['parse_failures']}
  {BOLD}API errors:{RESET}      {stats.get('api_errors', 0)}
  {BOLD}Fallbacks:{RESET}       {stats['fallbacks']}

  {BOLD}── Financial ──{RESET}
  Revenue:         ${total_revenue:.2f}
  Cost:            ${total_cost:.2f}
  {BOLD}Profit:          ${profit:.2f}{RESET}

  {BOLD}── Operations ──{RESET}
  Conversions:     {n_conversions}/{n_rounds} ({n_conversions/n_rounds:.0%})
  Matches:         {n_matches} ({n_matches/max(n_conversions,1):.0%} of converted)
  Reneged:         {n_reneged}
  Final queue:     {len(queue)}
""")

    if kpis_final:
        print(f"  {BOLD}── Agent KPIs (memory) ──{RESET}")
        print(f"  Avg price:     ${kpis_final['avg_price_quoted']:.2f}/mi")
        print(f"  Conversion:    {kpis_final['conversion_rate']:.0%}")
        print(f"  Match rate:    {kpis_final['match_rate']:.0%}")
        print(f"  Avg savings:   {kpis_final['avg_savings_when_matched']:.1%}")
        print(f"  Est. profit:   ${kpis_final['estimated_profit']:.2f}")
        print()

    print(f"  {BOLD}── Decision Trail ──{RESET}")
    for rec in agent.memory.history:
        conv = {True: "CONV", False: "REJ ", None: " ?  "}[rec["converted"]]
        match = ""
        if rec.get("matched") is True:
            match = f"MATCH(saved {rec.get('savings',0):.0%})"
        elif rec.get("matched") is False and rec.get("converted"):
            match = "WAIT"
        print(f"  R{rec['round']:>2}: ${rec['price']:.2f}  {conv} {match:20s} "
              f"q={rec['queue_size']:>2}  {rec['reasoning'][:42]}")

    print(f"\n  {BOLD}Results saved to:{RESET} {RESULTS_DIR}/")
    print(f"\n{DLINE}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    run_demo(args.rounds)
