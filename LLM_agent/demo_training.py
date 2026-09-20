#!/usr/bin/env python3
"""
Training Data Demo — runs the LLM agent on real riders from training_data.csv
with a proper sequential queue simulation.

Conversion rule:  The training data records each rider's quoted_price and
whether they converted.  We treat this as: the rider's WTP ≈ quoted_price
when they converted, and WTP < quoted_price when they didn't.
  • If training says convert=0 → always reject (WTP was already below market).
  • If training says convert=1 AND agent_price <= quoted_price → convert.
  • If training says convert=1 AND agent_price >  quoted_price → reject
    (agent priced above the WTP we observed in training).

Queue dynamics:
  • Converted + MATCH → both riders leave the queue.
  • Converted + WAIT  → rider enters the queue.
  • Riders renege after 180s of waiting (removed from queue before each round).

Usage:
    export OPENAI_API_KEY="sk-proj-..."
    python -m LLM_agent.demo_training                  # default 25 riders
    python -m LLM_agent.demo_training --riders 40      # more riders
    python -m LLM_agent.demo_training --week 2         # week 2 data

Output CSVs saved to LLM_agent/results_training/ (includes stage_coach_advisor.csv with
coach text + advisor JSON input). Defaults: decision and coach models gpt-5.4.
"""

import sys, os, time, csv, argparse, textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from rider import rider as Rider
from LLM_agent.config import PRICE_GRID, COST_PER_MILE, MAX_CANDIDATES
from LLM_agent.state_summary import summarize_state
from LLM_agent.prompt_builder import SYSTEM_PROMPT
from LLM_agent.agent_core import CalyberAgent

# ── display helpers ──────────────────────────────────────────────────

BLUE    = "\033[94m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

DLINE = f"{DIM}{'═' * 76}{RESET}"

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
    print(f"{MAGENTA}  └{'─' * 54}{RESET}")


def _csv_long_text(s, max_len=20000):
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n...[truncated]"


# ── CSV collector ────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_training")

class CSVCollector:
    def __init__(self):
        self.tables = {}

    def add(self, table, row):
        self.tables.setdefault(table, []).append(row)

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

# ── queue helpers ────────────────────────────────────────────────────

RENEGE_THRESHOLD = 180.0  # seconds before rider leaves queue

def renege_queue(queue, current_time):
    """Remove riders who've waited longer than the threshold."""
    before = len(queue)
    queue[:] = [r for r in queue if (current_time - r.arrival_time) < RENEGE_THRESHOLD]
    reneged = before - len(queue)
    return reneged

# ── main ─────────────────────────────────────────────────────────────

def run(n_riders, week):
    agent = CalyberAgent()
    if not agent._client:
        print(f"\n{RED}{BOLD}  ERROR: No OPENAI_API_KEY set!{RESET}")
        print(f"  Run: export OPENAI_API_KEY=\"sk-proj-...\"")
        sys.exit(1)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.csv")
    df = pd.read_csv(data_path)
    df = df[df.arrival_week == week].sort_values("arrival_time").head(n_riders).reset_index(drop=True)

    print(f"\n{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"{BOLD}{CYAN}   CALYBER LLM AGENT — TRAINING DATA SIMULATION{RESET}")
    print(f"{BOLD}{CYAN}   Week {week}, first {len(df)} riders{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 76}{RESET}")
    print(f"\n  Model: {agent._model}  |  Price grid: {PRICE_GRID}")
    print(f"  Renege threshold: {RENEGE_THRESHOLD}s  |  Cost/mile: ${COST_PER_MILE:.2f}")

    collector = CSVCollector()
    queue = []
    total_time = 0
    total_revenue = 0.0
    total_cost = 0.0
    n_conversions = 0
    n_matches = 0
    n_reneged_total = 0

    for idx, row in df.iterrows():
        round_num = idx + 1
        rider_obj = Rider(
            int(row.arrival_week), row.arrival_time,
            row.pickup_lat, row.pickup_lon,
            row.dropoff_lat, row.dropoff_lon,
            int(row.pickup_area), int(row.dropoff_area),
        )

        # Renege check
        reneged = renege_queue(queue, row.arrival_time)
        n_reneged_total += reneged
        if reneged:
            total_cost += sum(
                COST_PER_MILE * r.solo_length for r in []
            )

        train_price = row.quoted_price
        train_convert = int(row.convert_or_not)

        header(f"ROUND {round_num}/{len(df)}: rider #{int(row.rider_id)}  "
               f"area {int(row.pickup_area)}->{int(row.dropoff_area)}  "
               f"solo={row.solo_length:.2f}mi  t={row.arrival_time:.0f}s")

        if reneged:
            print(f"  {RED}  [{reneged} rider(s) reneged from queue]{RESET}")

        # ── STAGE 1: Perception ──────────────────────────────────────
        stage(1, "PERCEPTION — State Summarizer")
        summary = summarize_state(queue, rider_obj, max_candidates=MAX_CANDIDATES)
        r_info = summary["incoming_rider"]

        bullet("Incoming", f"area {r_info['pickup_area']}->{r_info['dropoff_area']}, "
                           f"{r_info['solo_length']} mi")
        bullet("Queue", f"{summary['queue_size']} riders "
                        f"(max wait: {summary['max_waiting_age']}s, avg: {summary['avg_waiting_age']}s)")

        collector.add("stage1_perception", {
            "round": round_num, "rider_id": int(row.rider_id),
            "arrival_time": row.arrival_time,
            "queue_size": summary["queue_size"],
            "pickup_area": r_info["pickup_area"],
            "dropoff_area": r_info["dropoff_area"],
            "solo_length": r_info["solo_length"],
            "max_waiting_age_sec": summary["max_waiting_age"],
            "avg_waiting_age_sec": summary["avg_waiting_age"],
            "num_candidates": len(summary["candidates"]),
            "reneged_this_round": reneged,
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
                    "round": round_num,
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

        mem_row = {"round": round_num, "rider_id": int(row.rider_id)}
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
        user_prompt = agent.build_user_prompt_for_round(summary, kpis, recent, rider_obj)
        pm = dict(agent.last_prompt_meta or {})
        collector.add("stage_coach_advisor", {
            "round": round_num,
            "rider_id": int(row.rider_id),
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
        decision = agent.decide(list(queue), rider_obj)
        elapsed_ms = (time.time() - t0) * 1000
        total_time += elapsed_ms

        bullet("Model", agent._model)
        bullet("Latency", f"{elapsed_ms:.0f}ms",
               GREEN if elapsed_ms < 3000 else YELLOW)
        bullet("Source", decision["source"],
               GREEN if decision["source"] == "llm" else RED)

        collector.add("stage4_llm_call", {
            "round": round_num, "rider_id": int(row.rider_id),
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

        # ── STAGE 5: Parser ──────────────────────────────────────────
        stage(5, "PARSER & VALIDATOR — Structured Output")
        agent_price = decision["price"]
        action = decision["decision"]
        cand_rank = decision.get("candidate_rank")

        bullet("Agent price", f"${agent_price:.2f}/mi", BOLD)
        bullet("Training price", f"${train_price:.3f}/mi "
               f"({'converted' if train_convert else 'rejected'})", DIM)
        bullet("Action", action, BOLD)
        if cand_rank:
            bullet("Candidate", f"#{cand_rank}")
        stage_end()

        # ── STAGE 6: Chain-of-Thought ────────────────────────────────
        stage(6, "CHAIN-OF-THOUGHT REASONING")
        reasoning = decision.get("reasoning", "(none)")
        wrapped = textwrap.fill(reasoning, width=60)
        for line in wrapped.split("\n"):
            print(f"{MAGENTA}  │{RESET}  {YELLOW}{BOLD}\"{line}\"{RESET}")
        stage_end()

        # ── OUTCOME — simulate conversion using training data ────────
        stage("*", "OUTCOME — Simulated from Training Data")

        if train_convert == 0:
            converts = False
            outcome_reason = "Training says rider rejected (WTP < training price)"
        elif agent_price <= train_price:
            converts = True
            outcome_reason = f"Agent price ${agent_price:.2f} <= training ${train_price:.3f}"
        else:
            converts = False
            outcome_reason = f"Agent price ${agent_price:.2f} > training ${train_price:.3f} (likely above WTP)"

        bullet("Conversion", f"{'YES' if converts else 'NO'}",
               GREEN if converts else RED)
        bullet("Reason", outcome_reason, DIM)

        matched_rider_obj = decision.get("matched_rider")
        matched_area = ""
        actual_savings = 0.0

        if converts:
            n_conversions += 1
            revenue = agent_price * rider_obj.solo_length
            total_revenue += revenue

            agent.memory.on_pricing(rider_obj, agent_price, decision,
                {"queue_size": len(queue), "candidates": summary["candidates"]})
            agent.memory.on_matching(rider_obj, decision,
                {"candidates": summary["candidates"]})

            if action == "MATCH" and matched_rider_obj is not None:
                n_matches += 1
                matched_area = f"{matched_rider_obj.pickup_area}->{matched_rider_obj.dropoff_area}"
                rank = decision.get("candidate_rank")
                cands = summary["candidates"]
                if rank and 1 <= rank <= len(cands):
                    actual_savings = cands[rank - 1]["savings_rate"]
                    trip_cost = COST_PER_MILE * cands[rank - 1]["trip_length"]
                else:
                    trip_cost = COST_PER_MILE * rider_obj.solo_length
                total_cost += trip_cost
                queue = [w for w in queue if w is not matched_rider_obj]
                bullet("Match", f"MATCHED with area {matched_area} (savings {actual_savings:.0%})", GREEN)
                bullet("Queue", f"{len(queue)} riders remaining")
            else:
                cost = COST_PER_MILE * rider_obj.solo_length
                total_cost += cost
                queue.append(rider_obj)
                bullet("Match", "WAIT — rider added to queue", CYAN)
                bullet("Queue", f"{len(queue)} riders now waiting")

            bullet("Revenue", f"${revenue:.2f}", GREEN)
        else:
            agent.memory.on_pricing(rider_obj, agent_price, decision,
                {"queue_size": len(queue), "candidates": summary["candidates"]})
            bullet("Revenue", "$0.00 (rejected)", RED)

        agent.pop_cache(rider_obj)

        collector.add("stage5_decisions", {
            "round": round_num, "rider_id": int(row.rider_id),
            "queue_size": summary["queue_size"],
            "agent_price": agent_price,
            "training_price": round(train_price, 4),
            "training_convert": train_convert,
            "action": action,
            "candidate_rank": cand_rank,
            "actual_converted": int(converts),
            "matched_rider_areas": matched_area,
            "savings_rate": round(actual_savings, 4) if actual_savings else "",
            "source": decision["source"],
            "revenue": round(agent_price * rider_obj.solo_length, 4) if converts else 0,
            "queue_after": len(queue),
        })

        collector.add("stage6_reasoning", {
            "round": round_num, "rider_id": int(row.rider_id),
            "reasoning": reasoning,
            "agent_price": agent_price,
            "action": action,
            "candidate_rank": cand_rank,
            "converted": int(converts),
            "source": decision["source"],
        })

        collector.add("master_summary", {
            "round": round_num,
            "rider_id": int(row.rider_id),
            "arrival_time": row.arrival_time,
            "pickup_area": int(row.pickup_area),
            "dropoff_area": int(row.dropoff_area),
            "solo_length": round(row.solo_length, 4),
            "queue_size_before": summary["queue_size"],
            "num_candidates": len(summary["candidates"]),
            "agent_price": agent_price,
            "training_price": round(train_price, 4),
            "training_convert": train_convert,
            "actual_converted": int(converts),
            "action": action,
            "candidate_rank": cand_rank,
            "matched_rider_areas": matched_area,
            "savings_rate": round(actual_savings, 4) if actual_savings else "",
            "reasoning": reasoning,
            "source": decision["source"],
            "latency_ms": round(elapsed_ms, 1),
            "revenue": round(agent_price * rider_obj.solo_length, 4) if converts else 0,
            "queue_size_after": len(queue),
            "reneged_this_round": reneged,
            "cumulative_revenue": round(total_revenue, 2),
            "cumulative_cost": round(total_cost, 2),
            "cumulative_profit": round(total_revenue - total_cost, 2),
            "kpi_conversion_rate": kpis["conversion_rate"] if kpis else None,
            "kpi_match_rate": kpis["match_rate"] if kpis else None,
            "kpi_est_profit": kpis["estimated_profit"] if kpis else None,
            "user_prompt": user_prompt,
            "compose_source": pm.get("source", ""),
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
    header("SIMULATION SUMMARY")
    stats = agent.stats
    profit = total_revenue - total_cost

    print(f"""
  {BOLD}Data source:{RESET}      training_data.csv (week {week}, {len(df)} riders)
  {BOLD}Total LLM time:{RESET}   {total_time:.0f}ms ({total_time/len(df):.0f}ms avg/round)
  {BOLD}LLM calls:{RESET}        {stats['llm_calls']}
  {BOLD}Parse failures:{RESET}   {stats['parse_failures']}
  {BOLD}API errors:{RESET}       {stats.get('api_errors', 0)}
  {BOLD}Fallbacks:{RESET}        {stats['fallbacks']}

  {BOLD}── Financial Summary ──{RESET}
  Revenue:          ${total_revenue:.2f}
  Cost:             ${total_cost:.2f}
  {BOLD}Profit:           ${profit:.2f}{RESET}  {'(' + GREEN + 'positive' + RESET + ')' if profit > 0 else '(' + RED + 'negative' + RESET + ')'}
  Profit/rider:     ${profit/len(df):.3f}

  {BOLD}── Operational Stats ──{RESET}
  Arrivals:         {len(df)}
  Conversions:      {n_conversions}  ({n_conversions/len(df):.0%})
  Matches:          {n_matches}   ({n_matches/max(n_conversions,1):.0%} of converted)
  Reneged:          {n_reneged_total}
  Final queue:      {len(queue)} riders
""")

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
    parser.add_argument("--riders", type=int, default=25)
    parser.add_argument("--week", type=int, default=1)
    args = parser.parse_args()
    run(args.riders, args.week)
