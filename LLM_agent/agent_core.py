"""
Agent Core — a *stateful* LLM agent that maintains memory across rounds,
tracks its own running KPIs, and feeds recent performance back into each
prompt so the model can genuinely adapt its strategy over time.

What makes this an agent (not just an LLM function call):
  • Memory     — rolling window of recent decisions and observed outcomes.
  • Adaptation — running KPIs (conversion rate, match rate, avg price) are
                 injected into every prompt so the LLM sees the consequences
                 of its own past actions and can course-correct.
  • Reasoning  — the LLM produces a "reasoning" field (chain-of-thought)
                 before each decision, making its logic inspectable.
  • Learning   — non-conversion events are detected automatically (if
                 matching_function is never called for a rider, the agent
                 infers the rider rejected the quoted price).
"""

import json
import time

from .config import (
    OPENAI_API_KEY, OPENROUTER_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    LLM_TIMEOUT, PRICE_GRID, MAX_CANDIDATES, COST_PER_MILE,
    MEMORY_WINDOW, RECENT_DISPLAY,
    ENABLE_LOGGING, LOG_FILE,
    ADVISOR_MODEL, ADVISOR_INTERVAL,
    DECISION_BACKEND, JEV_MODEL, OPENROUTER_DECISIONS_URL, OPENROUTER_TIMEOUT,
    NO_CANDIDATE_PRICE_FLOOR, LOW_PRICE_FLOOR, WEAK_MATCH_PRICE_FLOOR,
    STRONG_MATCH_SAVINGS, MODERATE_MATCH_SAVINGS, REJECT_WEAK_MATCH_SAVINGS,
    OLD_QUEUE_SEC, STALE_QUEUE_SEC,
    ALLOW_LOW_PRICE_CR, ALLOW_SUB60_CR,
    ALLOW_LOW_PRICE_WINDOW, ALLOW_SUB60_WINDOW,
    chat_completion_token_param,
)
from .state_summary import summarize_state
from .prompt_builder import SYSTEM_PROMPT, build_round_prompt
from .orchestrator_llm import compose_user_prompt
from .parser import parse_response, fallback_decision
from .jev_decision import query_jev


# ═══════════════════════════════════════════════════════════════════════
#  Agent Memory
# ═══════════════════════════════════════════════════════════════════════

class AgentMemory:
    """
    Rolling memory of decisions and outcomes.

    Each record tracks one rider arrival: the price quoted, whether the
    rider converted, whether we matched or waited, and the estimated
    savings.  Non-conversions are inferred retroactively: when a new
    pricing call arrives and the previous rider's record still shows
    converted=None, that rider must have rejected the price.
    """

    def __init__(self, window=MEMORY_WINDOW):
        self.window = window
        self.history = []

        self.total_arrivals = 0
        self.total_conversions = 0
        self.total_matches = 0
        self.total_waits = 0       # converted but chose WAIT
        self.total_solos_risk = 0  # non-conversions (rider rejected price)

        self._pending_rider_key = None
        self.price_buckets = {
            "low_<=0.55": self._empty_bucket(),
            "mid_0.60-0.70": self._empty_bucket(),
            "high_0.75-0.85": self._empty_bucket(),
            "premium_>=0.90": self._empty_bucket(),
        }
        self.context_buckets = {
            "no_candidates": self._empty_bucket(),
            "has_candidates": self._empty_bucket(),
            "strong_match_>=15%": self._empty_bucket(),
            "old_queue_>=45s": self._empty_bucket(),
        }

    # ── recording ────────────────────────────────────────────────────

    def on_pricing(self, rider, price, decision, summary):
        """Called every time pricing_function runs (= every arrival)."""
        self._close_pending_as_nonconversion()

        self.total_arrivals += 1
        self._pending_rider_key = id(rider)

        record = {
            "round": self.total_arrivals,
            "arrival_time": rider.arrival_time,
            "pickup_area": rider.pickup_area,
            "dropoff_area": rider.dropoff_area,
            "solo_length": round(rider.solo_length, 2),
            "price": price,
            "queue_size": summary["queue_size"],
            "n_candidates": len(summary["candidates"]),
            "max_waiting_age": summary.get("max_waiting_age", 0),
            "best_savings": self._best_savings(summary.get("candidates", [])),
            "action": decision.get("decision"),
            "candidate_rank": decision.get("candidate_rank"),
            "reasoning": decision.get("reasoning", ""),
            "source": decision.get("source", ""),
            "converted": None,
            "matched": None,
            "savings": None,
            "aggregated": False,
        }
        self.history.append(record)
        self._trim()

    def on_matching(self, rider, decision, summary):
        """
        Called every time matching_function runs.
        The simulator only calls matching after conversion, so reaching
        here means the rider accepted the quoted price.
        """
        self.total_conversions += 1
        self._pending_rider_key = None

        rec = self._last_record()
        if rec is None:
            return
        rec["converted"] = True

        if decision.get("decision") == "MATCH" and decision.get("matched_rider") is not None:
            rec["matched"] = True
            self.total_matches += 1
            rank = decision.get("candidate_rank")
            cands = summary.get("candidates", [])
            if rank and 1 <= rank <= len(cands):
                rec["savings"] = cands[rank - 1]["savings_rate"]
        else:
            rec["matched"] = False
            self.total_waits += 1
        self._aggregate_record(rec)

    # ── KPI computation ──────────────────────────────────────────────

    def get_running_kpis(self):
        """Compute KPIs over the recent window (and lifetime totals)."""
        recent = self.history[-self.window:]
        if not recent:
            return None

        n = len(recent)
        converted = [r for r in recent if r["converted"] is True]
        rejected = [r for r in recent if r["converted"] is False]
        matched = [r for r in recent if r["matched"] is True]
        waited = [r for r in recent if r["matched"] is False]

        avg_price = sum(r["price"] for r in recent) / n
        conversion_rate = len(converted) / n if n else 0
        match_rate = len(matched) / len(converted) if converted else 0

        savings_vals = [r["savings"] for r in matched if r["savings"] is not None]
        avg_savings = sum(savings_vals) / len(savings_vals) if savings_vals else 0

        revenue_est = sum(
            r["price"] * r["solo_length"] for r in converted
        )
        cost_est = sum(
            COST_PER_MILE * r["solo_length"] * (1 - (r["savings"] or 0))
            for r in converted
        )

        return {
            "window_size": n,
            "avg_price_quoted": round(avg_price, 3),
            "conversion_rate": round(conversion_rate, 3),
            "match_rate": round(match_rate, 3),
            "avg_savings_when_matched": round(avg_savings, 4),
            "estimated_revenue": round(revenue_est, 2),
            "estimated_cost": round(cost_est, 2),
            "estimated_profit": round(revenue_est - cost_est, 2),
            "estimated_profit_per_arrival": round((revenue_est - cost_est) / n, 3) if n else 0,
            "estimated_profit_per_conversion": (
                round((revenue_est - cost_est) / len(converted), 3) if converted else 0
            ),
            "lifetime_arrivals": self.total_arrivals,
            "lifetime_conversions": self.total_conversions,
            "lifetime_matches": self.total_matches,
            "aggregate_memory": self._aggregate_summary(),
        }

    def get_recent_decisions(self, n=RECENT_DISPLAY):
        """Return the last *n* records formatted for prompt injection."""
        return self.history[-n:]

    # ── internals ────────────────────────────────────────────────────

    def _close_pending_as_nonconversion(self):
        """
        If the previous rider's record still shows converted=None, that
        rider did not convert (matching_function was never called for them).
        """
        if self._pending_rider_key is not None and self.history:
            rec = self._last_record()
            if rec and rec["converted"] is None:
                rec["converted"] = False
                rec["matched"] = False
                self.total_solos_risk += 1
                self._aggregate_record(rec)
        self._pending_rider_key = None

    def _last_record(self):
        return self.history[-1] if self.history else None

    def _trim(self):
        if len(self.history) > self.window * 3:
            self.history = self.history[-self.window * 2:]

    def reset_week(self):
        """Called between simulation weeks to keep KPIs per-week clean."""
        self.history.clear()
        self._pending_rider_key = None

    @staticmethod
    def _empty_bucket():
        return {
            "n": 0,
            "conversions": 0,
            "matches": 0,
            "revenue": 0.0,
            "cost": 0.0,
        }

    @staticmethod
    def _best_savings(candidates):
        if not candidates:
            return 0
        return max((c.get("savings_rate", 0) or 0) for c in candidates)

    @staticmethod
    def _price_bucket(price):
        if price <= 0.55:
            return "low_<=0.55"
        if price <= 0.70:
            return "mid_0.60-0.70"
        if price <= 0.85:
            return "high_0.75-0.85"
        return "premium_>=0.90"

    def _context_keys(self, rec):
        keys = ["has_candidates" if rec.get("n_candidates", 0) else "no_candidates"]
        if (rec.get("best_savings") or 0) >= STRONG_MATCH_SAVINGS:
            keys.append("strong_match_>=15%")
        if (rec.get("max_waiting_age") or 0) >= OLD_QUEUE_SEC:
            keys.append("old_queue_>=45s")
        return keys

    def _update_bucket(self, bucket, rec):
        bucket["n"] += 1
        if rec.get("converted") is True:
            bucket["conversions"] += 1
            if rec.get("matched") is True:
                bucket["matches"] += 1
            revenue = rec["price"] * rec["solo_length"]
            cost = COST_PER_MILE * rec["solo_length"] * (1 - (rec.get("savings") or 0))
            bucket["revenue"] += revenue
            bucket["cost"] += cost

    def _aggregate_record(self, rec):
        """Store stable lifetime summaries without bloating the prompt with raw rows."""
        if rec.get("aggregated"):
            return
        if rec.get("converted") is None:
            return

        self._update_bucket(self.price_buckets[self._price_bucket(rec["price"])], rec)
        for key in self._context_keys(rec):
            self._update_bucket(self.context_buckets[key], rec)
        rec["aggregated"] = True

    @staticmethod
    def _summarize_bucket(bucket):
        n = bucket["n"]
        if not n:
            return None
        conversions = bucket["conversions"]
        return {
            "n": n,
            "conversion_rate": round(conversions / n, 3),
            "match_rate": round(bucket["matches"] / conversions, 3) if conversions else 0,
            "profit": round(bucket["revenue"] - bucket["cost"], 2),
        }

    def _aggregate_summary(self):
        return {
            "price_buckets": {
                k: v for k, v in (
                    (name, self._summarize_bucket(bucket))
                    for name, bucket in self.price_buckets.items()
                ) if v
            },
            "context_buckets": {
                k: v for k, v in (
                    (name, self._summarize_bucket(bucket))
                    for name, bucket in self.context_buckets.items()
                ) if v
            },
        }


# ═══════════════════════════════════════════════════════════════════════
#  Calyber Agent
# ═══════════════════════════════════════════════════════════════════════

class CalyberAgent:
    """
    Stateful LLM agent with memory, adaptation, and chain-of-thought.
    """

    def __init__(
        self,
        api_key=None,
        model=None,
        advisor_model=None,
        orchestrator_model=None,
        decision_backend=None,
        openrouter_api_key=None,
        jev_model=None,
    ):
        self._api_key = api_key or OPENAI_API_KEY
        self._decision_backend = (decision_backend or DECISION_BACKEND).lower()
        if self._decision_backend not in ("jev", "openai"):
            raise ValueError("decision_backend must be 'jev' or 'openai'")
        self._openai_model = model or MODEL_NAME
        self._jev_model = jev_model or JEV_MODEL
        self._model = self._jev_model if self._decision_backend == "jev" else self._openai_model
        self._openrouter_api_key = openrouter_api_key or OPENROUTER_API_KEY
        if advisor_model is not None:
            self._advisor_model = advisor_model
        elif orchestrator_model is not None:
            self._advisor_model = orchestrator_model
        else:
            self._advisor_model = ADVISOR_MODEL
        self._client = None
        self._cache = {}
        self.memory = AgentMemory(window=MEMORY_WINDOW)
        self._last_user_prompt = None
        self._last_prompt_rider_id = None
        self._last_prompt_meta = None
        self._cached_advisor_text = ""
        self._advisor_last_refresh_round = None
        self._stats = {
            "total_calls": 0,
            "llm_calls": 0,
            "advisor_calls": 0,
            "advisor_latency_ms": 0.0,
            "cache_hits": 0,
            "fallbacks": 0,
            "parse_failures": 0,
            "api_errors": 0,
            "total_latency_ms": 0.0,
        }
        self._log_fh = None

        if self._api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    timeout=LLM_TIMEOUT,
                )
            except ImportError:
                print("[LLM_Agent] openai package not installed — running in fallback-only mode.")
            except Exception as e:
                print(f"[LLM_Agent] Failed to init OpenAI client: {e}")
        else:
            print("[LLM_Agent] No OPENAI_API_KEY set — strategic advisor is disabled.")

        if self._decision_backend == "jev" and not self._openrouter_api_key:
            print("[LLM_Agent] No OPENROUTER_API_KEY set — Jev decisions will use fallback.")

    # ── public API (called by policy wrapper) ───────────────────────

    def decide(self, state, rider):
        """
        Return a decision dict for the given (state, rider) pair.

        On the first call for a rider (from pricing_function):
          - Summarizes state
          - Queries the LLM with memory context
          - Parses, validates, caches
          - Records in memory

        On the second call (from matching_function):
          - Returns the cached decision
        """
        self._stats["total_calls"] += 1
        rider_key = id(rider)

        if rider_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[rider_key]

        summary = summarize_state(state, rider, max_candidates=MAX_CANDIDATES)
        kpis = self.memory.get_running_kpis()
        recent = self.memory.get_recent_decisions()

        if self._last_prompt_rider_id != rider_key or not self._last_user_prompt:
            self.build_user_prompt_for_round(summary, kpis, recent, rider)

        try:
            decision = self._query_llm(summary, kpis, recent)
            decision = self._resolve_match(decision, summary)
            decision = self._apply_profit_guardrails(decision, summary, kpis)
            decision["_state_snapshot"] = {
                "queue_size": summary["queue_size"],
                "max_waiting_age": summary["max_waiting_age"],
                "candidates": summary["candidates"],
            }

            self._cache[rider_key] = decision
            self._log_decision(summary, decision, kpis)

            return decision
        finally:
            self._last_user_prompt = None
            self._last_prompt_rider_id = None
            self._last_prompt_meta = None

    def record_pricing(self, rider, decision, state):
        """Called by LLMPricingPolicy after decide()."""
        snapshot = decision.get("_state_snapshot") or {}
        summary_lite = {
            "queue_size": snapshot.get("queue_size", len(state)),
            "max_waiting_age": snapshot.get("max_waiting_age", 0),
            "candidates": snapshot.get("candidates", []),
        }
        self.memory.on_pricing(rider, decision["price"], decision, summary_lite)

    def record_matching(self, rider, decision, state):
        """Called by LLMMatchingPolicy after decide()."""
        summary_lite = {
            "candidates": decision.get("_candidates_snapshot", []),
        }
        self.memory.on_matching(rider, decision, summary_lite)

    def pop_cache(self, rider):
        """Remove and return cached decision."""
        return self._cache.pop(id(rider), None)

    @property
    def stats(self):
        return dict(self._stats)

    @property
    def last_prompt_meta(self):
        """Metadata from the last compose_user_prompt (coach text, advisor JSON, errors)."""
        return self._last_prompt_meta

    def build_user_prompt_for_round(self, summary, kpis, recent, rider):
        """
        Compose the decision agent's user message (template, plus optional coach advice).
        Call before decide() from demos to print the same prompt the agent uses.
        """
        round_number = self.memory.total_arrivals + 1
        should_refresh = (
            bool(self._advisor_model)
            and self._client is not None
            and (
                self._advisor_last_refresh_round is None
                or round_number - self._advisor_last_refresh_round >= ADVISOR_INTERVAL
            )
        )
        cached_advice = None if should_refresh else self._cached_advisor_text
        self._last_user_prompt, self._last_prompt_meta = compose_user_prompt(
            self._client,
            self._advisor_model,
            summary,
            kpis,
            recent,
            PRICE_GRID,
            cached_advice=cached_advice,
            skip_advisor=not should_refresh and not bool(cached_advice),
        )
        self._last_prompt_rider_id = id(rider)
        meta = self._last_prompt_meta or {}
        if meta.get("source") == "template+advisor":
            self._stats["advisor_calls"] += 1
            self._stats["advisor_latency_ms"] += float(meta.get("advisor_ms") or 0)
            self._cached_advisor_text = meta.get("coach_advice", "")
            self._advisor_last_refresh_round = round_number
        elif should_refresh and meta.get("advisor_error"):
            self._advisor_last_refresh_round = round_number
        return self._last_user_prompt

    # ── LLM interaction ─────────────────────────────────────────────

    def _query_llm(self, summary, kpis, recent):
        if self._decision_backend == "jev":
            return self._query_jev(summary, kpis, recent)

        if self._client is None:
            self._stats["fallbacks"] += 1
            fb = fallback_decision(summary)
            fb["source"] = "fallback"
            fb["reasoning"] = "no LLM client"
            fb["_candidates_snapshot"] = summary["candidates"]
            return fb

        user_prompt = self._last_user_prompt or build_round_prompt(
            summary, kpis, recent, price_grid=PRICE_GRID
        )
        num_candidates = len(summary["candidates"])

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                **chat_completion_token_param(self._model, MAX_TOKENS),
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as e:
            self._stats["api_errors"] += 1
            self._stats["fallbacks"] += 1
            print(f"[LLM_Agent] API error: {e}")
            fb = fallback_decision(summary)
            fb["source"] = "fallback"
            fb["reasoning"] = f"API error: {e}"
            fb["_candidates_snapshot"] = summary["candidates"]
            return fb
        finally:
            elapsed = (time.time() - t0) * 1000
            self._stats["total_latency_ms"] += elapsed
            self._stats["llm_calls"] += 1

        parsed = parse_response(raw_text, PRICE_GRID, num_candidates)
        if parsed is None:
            self._stats["parse_failures"] += 1
            self._stats["fallbacks"] += 1
            fb = fallback_decision(summary)
            fb["source"] = "fallback"
            fb["reasoning"] = f"parse failure on: {raw_text[:100]}"
            fb["raw"] = raw_text
            fb["_candidates_snapshot"] = summary["candidates"]
            return fb

        parsed["source"] = "llm"
        parsed["_candidates_snapshot"] = summary["candidates"]
        return parsed

    def _query_jev(self, summary, kpis, recent):
        if not self._openrouter_api_key:
            self._stats["fallbacks"] += 1
            fb = fallback_decision(summary)
            fb["source"] = "fallback"
            fb["reasoning"] = "no OpenRouter API key"
            fb["_candidates_snapshot"] = summary["candidates"]
            return fb

        advisor_text = (self._last_prompt_meta or {}).get("coach_advice", "")
        t0 = time.time()
        try:
            decision = query_jev(
                api_key=self._openrouter_api_key,
                endpoint=OPENROUTER_DECISIONS_URL,
                model=self._jev_model,
                summary=summary,
                kpis=kpis,
                recent=recent,
                price_grid=PRICE_GRID,
                advisor_text=advisor_text,
                timeout=OPENROUTER_TIMEOUT,
            )
            decision["source"] = "jev"
            decision["_candidates_snapshot"] = summary["candidates"]
            return decision
        except Exception as e:
            self._stats["api_errors"] += 1
            self._stats["fallbacks"] += 1
            print(f"[LLM_Agent] Jev API error: {e}")
            fb = fallback_decision(summary)
            fb["source"] = "fallback"
            fb["reasoning"] = f"Jev API error: {e}"
            fb["_candidates_snapshot"] = summary["candidates"]
            return fb
        finally:
            elapsed = (time.time() - t0) * 1000
            self._stats["total_latency_ms"] += elapsed
            self._stats["llm_calls"] += 1

    # ── helpers ──────────────────────────────────────────────────────

    def _resolve_match(self, decision, summary):
        candidates = summary["candidates"]
        if decision["decision"] == "MATCH" and decision.get("candidate_rank"):
            rank = decision["candidate_rank"]
            if 1 <= rank <= len(candidates):
                decision["matched_rider"] = candidates[rank - 1]["rider_ref"]
            else:
                decision["decision"] = "WAIT"
                decision["candidate_rank"] = None
                decision["matched_rider"] = None
        else:
            decision["matched_rider"] = None
        return decision

    def _apply_profit_guardrails(self, decision, summary, kpis):
        """
        Enforce conservative profit rules after parsing.

        The prompt should usually guide the LLM there, but guardrails prevent
        the most costly validation patterns: cheap no-match conversions and
        waiting on aged, reasonable-savings queues.
        """
        candidates = summary["candidates"]
        best = candidates[0] if candidates else None
        best_savings = (best or {}).get("savings_rate", 0) or 0
        max_wait = summary.get("max_waiting_age", 0) or 0
        cr = (kpis or {}).get("conversion_rate", 0.4)
        window = (kpis or {}).get("window_size", 0)

        notes = []

        # No immediate savings path: avoid buying low-margin conversions.
        if not candidates and decision["price"] < NO_CANDIDATE_PRICE_FLOOR:
            if not (window >= ALLOW_LOW_PRICE_WINDOW and cr < ALLOW_LOW_PRICE_CR):
                decision["price"] = NO_CANDIDATE_PRICE_FLOOR
                notes.append("guardrail: raised no-candidate price")

        # Low prices are only justified by strong savings or severe conversion failure.
        if decision["price"] < LOW_PRICE_FLOOR:
            if not (
                best_savings >= STRONG_MATCH_SAVINGS
                and window >= ALLOW_SUB60_WINDOW
                and cr < ALLOW_SUB60_CR
            ):
                decision["price"] = LOW_PRICE_FLOOR
                notes.append("guardrail: raised sub-0.60 quote")

        # If candidates are weak, do not subsidize conversion too much.
        if (
            candidates
            and best_savings < MODERATE_MATCH_SAVINGS
            and decision["price"] < WEAK_MATCH_PRICE_FLOOR
        ):
            decision["price"] = WEAK_MATCH_PRICE_FLOOR
            notes.append("guardrail: raised weak-match price")

        # Old queues should be cleared when a moderate-savings match exists.
        if (
            candidates
            and decision["decision"] == "WAIT"
            and max_wait >= OLD_QUEUE_SEC
            and best_savings >= MODERATE_MATCH_SAVINGS
        ):
            decision["decision"] = "MATCH"
            decision["candidate_rank"] = best["rank"]
            decision["matched_rider"] = best["rider_ref"]
            notes.append("guardrail: matched old queue")

        # Do not execute a match below the minimum savings threshold unless the queue is stale.
        if (
            decision["decision"] == "MATCH"
            and decision.get("candidate_rank")
            and candidates
        ):
            chosen = candidates[decision["candidate_rank"] - 1]
            if chosen.get("savings_rate", 0) < REJECT_WEAK_MATCH_SAVINGS and max_wait < STALE_QUEUE_SEC:
                decision["decision"] = "WAIT"
                decision["candidate_rank"] = None
                decision["matched_rider"] = None
                notes.append("guardrail: rejected weak match")

        if notes:
            base = decision.get("reasoning") or ""
            decision["reasoning"] = (base + " " + "; ".join(notes)).strip()[:160]
        return decision

    def _log_decision(self, summary, decision, kpis):
        if not ENABLE_LOGGING:
            return
        try:
            if self._log_fh is None:
                self._log_fh = open(LOG_FILE, "a")

            record = {
                "ts": time.time(),
                "round": self.memory.total_arrivals + 1,
                "rider": summary["incoming_rider"],
                "queue_size": summary["queue_size"],
                "n_candidates": len(summary["candidates"]),
                "price": decision.get("price"),
                "decision": decision.get("decision"),
                "candidate_rank": decision.get("candidate_rank"),
                "reasoning": decision.get("reasoning", ""),
                "source": decision.get("source"),
                "kpis": kpis,
            }
            self._log_fh.write(json.dumps(record) + "\n")
            self._log_fh.flush()
        except Exception:
            pass

    def __del__(self):
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception:
                pass


# ── singleton ───────────────────────────────────────────────────────

_shared_agent = None


def get_shared_agent(**kwargs):
    global _shared_agent
    if _shared_agent is None:
        _shared_agent = CalyberAgent(**kwargs)
    return _shared_agent
