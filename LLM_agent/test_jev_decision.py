import unittest

from .jev_decision import build_jev_request, parse_jev_response


PRICE_GRID = [0.35, 0.40, 0.75, 0.90]


class JevDecisionTests(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "incoming_rider": {
                "pickup_area": 1,
                "dropoff_area": 2,
                "solo_length": 4.0,
                "arrival_time": 100.0,
            },
            "queue_size": 2,
            "max_waiting_age": 55.0,
            "avg_waiting_age": 30.0,
            "candidates": [
                {
                    "rank": 1,
                    "savings_rate": 0.18,
                    "cost_if_matched": 0.58,
                    "waiting_age_sec": 55.0,
                    "rider_ref": object(),
                }
            ],
        }

    def test_request_uses_typed_choices_and_removes_object_references(self):
        request = build_jev_request(
            "typesafe/jev-1.13",
            self.summary,
            {"conversion_rate": 0.4},
            [],
            PRICE_GRID,
            advisor_text="Protect margin while clearing old, strong matches.",
        )

        self.assertEqual(request["questions"]["price"]["type"], "choice")
        self.assertEqual(request["questions"]["match"]["type"], "choice")
        self.assertIn("price_75", request["questions"]["price"]["criteria"])
        self.assertIn("match_1", request["questions"]["match"]["criteria"])
        candidate = request["state"]["current_state"]["candidates"][0]
        self.assertNotIn("rider_ref", candidate)

    def test_response_maps_to_calyber_decision(self):
        response = {
            "answers": {
                "price": {
                    "type": "choice",
                    "choice": "price_75",
                    "confidence": 0.84,
                    "probabilities": {"price_75": 0.84},
                },
                "match": {
                    "type": "choice",
                    "choice": "match_1",
                    "confidence": 0.91,
                    "probabilities": {"match_1": 0.91},
                },
            }
        }

        decision = parse_jev_response(response, PRICE_GRID, num_candidates=1)
        self.assertEqual(decision["price"], 0.75)
        self.assertEqual(decision["decision"], "MATCH")
        self.assertEqual(decision["candidate_rank"], 1)
        self.assertIn("price confidence 0.84", decision["reasoning"])

    def test_invalid_candidate_is_rejected(self):
        response = {
            "answers": {
                "price": {"choice": "price_75"},
                "match": {"choice": "match_2"},
            }
        }
        with self.assertRaises(ValueError):
            parse_jev_response(response, PRICE_GRID, num_candidates=1)


if __name__ == "__main__":
    unittest.main()
