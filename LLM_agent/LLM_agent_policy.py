"""
LLM Agent Policy — drop-in Calyber policy classes backed by a live,
stateful decision agent with memory and adaptation.

The agent:
  • Remembers its recent decisions and their outcomes
  • Tracks running KPIs (conversion rate, match rate, profit estimate)
  • Feeds that history back into every prompt so the LLM can adapt
  • Uses typed Jev decisions or an OpenAI chat-model fallback

Usage::

    from LLM_agent import LLMPricingPolicy, LLMMatchingPolicy
    simulator.simulation(LLMPricingPolicy(), LLMMatchingPolicy(), exe_time_limit=30)
"""

from .config import TEAM_NAME
from .agent_core import get_shared_agent


class LLMPricingPolicy:

    def __init__(self, c=0.70):
        self.c = c
        self._agent = get_shared_agent()

    @staticmethod
    def get_name():
        return TEAM_NAME

    def pricing_function(self, state, rider):
        """
        Quote a price for the incoming rider.

        This triggers the full agentic pipeline:
          1. Check memory for running KPIs and recent decisions
          2. Summarize current state (queue, candidates)
          3. Build prompt with KPIs + history + state
          4. Call the configured model for price and match decisions
          5. Parse, validate, cache
          6. Record this pricing event in memory
        """
        decision = self._agent.decide(state, rider)
        self._agent.record_pricing(rider, decision, state)
        return decision["price"]


class LLMMatchingPolicy:

    def __init__(self, c=0.70):
        self.c = c
        self._agent = get_shared_agent()

    @staticmethod
    def get_name():
        return TEAM_NAME

    def matching_function(self, state, rider):
        """
        Decide whether to match the incoming (converted) rider.

        The simulator only calls this if the rider converted — so reaching
        here means the agent's price was accepted.  The agent records this
        conversion in memory, updating its running KPIs for future rounds.
        """
        decision = self._agent.decide(state, rider)

        matched = decision.get("matched_rider")

        if matched is not None and matched not in state:
            matched = None
            decision["matched_rider"] = None
            decision["decision"] = "WAIT"

        self._agent.record_matching(rider, decision, state)
        self._agent.pop_cache(rider)
        return matched
