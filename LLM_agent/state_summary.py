"""
State Summarizer — reads the current simulator state and incoming rider,
produces a compact structured snapshot suitable for LLM prompting.

Computes:
  - incoming rider features
  - queue statistics
  - top-K candidate matches ranked by savings rate
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import populate_shared_ride_lengths

from .config import MAX_CANDIDATES, COST_PER_MILE


def _rider_coords(r):
    return (r.pickup_lat, r.pickup_lon), (r.dropoff_lat, r.dropoff_lon)


def compute_candidates(state, rider, max_candidates=MAX_CANDIDATES):
    """
    For each waiting rider in *state*, compute shared-ride metrics against
    the incoming *rider*.  Return a list of candidate dicts sorted by
    savings_rate descending, capped at *max_candidates*.

    Each candidate dict carries:
        rider_ref        – the actual rider object (for returning to simulator)
        rank             – 1-indexed rank among candidates
        savings_rate     – (solo_i + solo_j − shared_trip) / (solo_i + solo_j)
        shared_length    – miles of overlapping segment
        trip_length      – total shared-route miles
        i_solo_length    – incoming rider's solo portion in the shared route
        j_solo_length    – waiting rider's solo portion in the shared route
        cost_if_matched  – estimated cost/mile attributed to incoming rider
        waiting_age_sec  – approx seconds the waiting rider has been waiting
        pickup_area      – waiting rider's pickup area id
        dropoff_area     – waiting rider's dropoff area id
    """
    origin_i, dest_i = _rider_coords(rider)
    candidates = []

    for waiting_rider in state:
        origin_j, dest_j = _rider_coords(waiting_rider)

        trip_length, shared_length, i_solo, j_solo, _ = populate_shared_ride_lengths(
            origin_i, dest_i, origin_j, dest_j
        )

        if shared_length <= 0:
            continue

        solo_sum = rider.solo_length + waiting_rider.solo_length
        if solo_sum <= 0:
            continue

        savings_rate = (solo_sum - trip_length) / solo_sum

        if savings_rate <= 0:
            continue

        cost_i_matched = COST_PER_MILE * (i_solo + shared_length / 2) / rider.solo_length if rider.solo_length > 0 else COST_PER_MILE

        waiting_age = max(0.0, rider.arrival_time - waiting_rider.arrival_time)

        candidates.append({
            "rider_ref": waiting_rider,
            "savings_rate": round(savings_rate, 4),
            "shared_length": round(shared_length, 2),
            "trip_length": round(trip_length, 2),
            "i_solo_length": round(i_solo, 2),
            "j_solo_length": round(j_solo, 2),
            "cost_if_matched": round(cost_i_matched, 4),
            "waiting_age_sec": round(waiting_age, 1),
            "pickup_area": waiting_rider.pickup_area,
            "dropoff_area": waiting_rider.dropoff_area,
        })

    candidates.sort(key=lambda x: x["savings_rate"], reverse=True)
    top = candidates[:max_candidates]
    for i, c in enumerate(top):
        c["rank"] = i + 1
    return top


def summarize_state(state, rider, max_candidates=MAX_CANDIDATES):
    """
    Build a complete state snapshot for one arrival round.

    Returns a dict with:
        incoming_rider  – dict of rider features
        queue_size      – int
        num_compatible  – int (candidates with positive savings before cap)
        candidates      – list of candidate dicts (at most max_candidates)
    """
    candidates = compute_candidates(state, rider, max_candidates=max_candidates)

    waiting_ages = [
        max(0.0, rider.arrival_time - w.arrival_time) for w in state
    ] if state else []

    return {
        "incoming_rider": {
            "pickup_area": rider.pickup_area,
            "dropoff_area": rider.dropoff_area,
            "solo_length": round(rider.solo_length, 2),
            "arrival_time": round(rider.arrival_time, 1),
        },
        "queue_size": len(state),
        "max_waiting_age": round(max(waiting_ages), 1) if waiting_ages else 0.0,
        "avg_waiting_age": round(sum(waiting_ages) / len(waiting_ages), 1) if waiting_ages else 0.0,
        "candidates": candidates,
    }
