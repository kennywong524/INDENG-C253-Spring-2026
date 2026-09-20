import os

TEAM_NAME = "LLM_Agent"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Jev is a decisions model and uses OpenRouter's dedicated Decisions endpoint,
# not the OpenAI-compatible chat endpoint.
DECISION_BACKEND = os.environ.get("CALYBER_DECISION_BACKEND", "jev").strip().lower()
JEV_MODEL = os.environ.get("CALYBER_JEV_MODEL", "typesafe/jev-1.13")
OPENROUTER_DECISIONS_URL = os.environ.get(
    "CALYBER_OPENROUTER_DECISIONS_URL",
    "https://openrouter.ai/api/alpha/decisions",
)
OPENROUTER_TIMEOUT = float(os.environ.get("CALYBER_OPENROUTER_TIMEOUT", "10.0"))

# Used only when CALYBER_DECISION_BACKEND=openai.
MODEL_NAME = os.environ.get("CALYBER_LLM_MODEL", "gpt-5.4")
TEMPERATURE = float(os.environ.get("CALYBER_LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.environ.get("CALYBER_LLM_MAX_TOKENS", "150"))
LLM_TIMEOUT = float(os.environ.get("CALYBER_LLM_TIMEOUT", "15.0"))

# Optional coach/advisor model: analyzes conversion & match tradeoffs only; output is
# appended after build_round_prompt. Does NOT replace the template.
# CALYBER_ORCHESTRATOR_MODEL still works as an alias for backward compatibility.
def _advisor_model_env():
    """Default coach model gpt-5.4; set CALYBER_ADVISOR_MODEL= to empty to disable."""
    v = os.environ.get("CALYBER_ADVISOR_MODEL")
    if v is not None:
        return v
    v = os.environ.get("CALYBER_ORCHESTRATOR_MODEL")
    if v is not None:
        return v
    return "gpt-5.4"


ADVISOR_MODEL = _advisor_model_env()
ORCHESTRATOR_MODEL = ADVISOR_MODEL  # alias
ADVISOR_INTERVAL = max(1, int(os.environ.get("CALYBER_ADVISOR_INTERVAL", "25")))

def _adv_float(primary, legacy, default):
    v = os.environ.get(primary)
    if v is not None:
        return float(v)
    v = os.environ.get(legacy)
    if v is not None:
        return float(v)
    return float(default)


def _adv_int(primary, legacy, default):
    v = os.environ.get(primary)
    if v is not None:
        return int(v)
    v = os.environ.get(legacy)
    if v is not None:
        return int(v)
    return int(default)


ADVISOR_TEMPERATURE = _adv_float("CALYBER_ADVISOR_TEMPERATURE", "CALYBER_ORCHESTRATOR_TEMPERATURE", "0.3")
ADVISOR_MAX_TOKENS = _adv_int("CALYBER_ADVISOR_MAX_TOKENS", "CALYBER_ORCHESTRATOR_MAX_TOKENS", "500")
ORCHESTRATOR_TEMPERATURE = ADVISOR_TEMPERATURE
ORCHESTRATOR_MAX_TOKENS = ADVISOR_MAX_TOKENS

COST_PER_MILE = 0.70

PRICE_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

MAX_CANDIDATES = 3

FALLBACK_PRICE = 0.75
MIN_SAVINGS_THRESHOLD = 0.05

# ── Agent memory ────────────────────────────────────────────────────────
MEMORY_WINDOW = int(os.environ.get("CALYBER_MEMORY_WINDOW", "50"))
RECENT_DISPLAY = int(os.environ.get("CALYBER_RECENT_DISPLAY", "8"))

# ── Profit guardrails / ablation knobs ───────────────────────────────────────
NO_CANDIDATE_PRICE_FLOOR = float(os.environ.get("CALYBER_NO_CANDIDATE_PRICE_FLOOR", "0.75"))
LOW_PRICE_FLOOR = float(os.environ.get("CALYBER_LOW_PRICE_FLOOR", "0.60"))
WEAK_MATCH_PRICE_FLOOR = float(os.environ.get("CALYBER_WEAK_MATCH_PRICE_FLOOR", "0.70"))

STRONG_MATCH_SAVINGS = float(os.environ.get("CALYBER_STRONG_MATCH_SAVINGS", "0.15"))
MODERATE_MATCH_SAVINGS = float(os.environ.get("CALYBER_MODERATE_MATCH_SAVINGS", "0.08"))
REJECT_WEAK_MATCH_SAVINGS = float(os.environ.get("CALYBER_REJECT_WEAK_MATCH_SAVINGS", "0.05"))

OLD_QUEUE_SEC = float(os.environ.get("CALYBER_OLD_QUEUE_SEC", "45"))
STALE_QUEUE_SEC = float(os.environ.get("CALYBER_STALE_QUEUE_SEC", "60"))

ALLOW_LOW_PRICE_CR = float(os.environ.get("CALYBER_ALLOW_LOW_PRICE_CR", "0.30"))
ALLOW_SUB60_CR = float(os.environ.get("CALYBER_ALLOW_SUB60_CR", "0.35"))
ALLOW_LOW_PRICE_WINDOW = int(os.environ.get("CALYBER_ALLOW_LOW_PRICE_WINDOW", "20"))
ALLOW_SUB60_WINDOW = int(os.environ.get("CALYBER_ALLOW_SUB60_WINDOW", "20"))

ENABLE_LOGGING = os.environ.get("CALYBER_LLM_LOG", "1") == "1"
LOG_FILE = os.environ.get("CALYBER_LLM_LOG_FILE", "llm_agent_decisions.jsonl")


def chat_completion_token_param(model_name: str, max_output: int) -> dict:
    """
    GPT-5+ and some OpenAI models reject `max_tokens` on chat.completions;
    they require `max_completion_tokens` instead.
    """
    name = (model_name or "").lower()
    if (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    ):
        return {"max_completion_tokens": max_output}
    return {"max_tokens": max_output}
