"""protocol forks: output differences judged as near-ties in quality mode speed."""
import json
from pathlib import Path
import unittest

from crit_eval.reports import status
from crit_eval.run import forks

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "crit_eval/protocol.json").read_text(encoding="utf-8"))


class Scorer:
    """An engine whose scoring path gives each continuation position the same alternatives."""
    def __init__(self, top):
        self.top, self.calls = top, []

    def score(self, prompt, continuation, top_k, prompt_rows=True):
        self.calls.append((list(prompt), list(continuation), top_k, prompt_rows))
        rows = [{"logprob": dict(self.top).get(t, -30.0), "top": [list(p) for p in self.top]} for t in continuation]
        return {"prompt": [], "continuation": rows}


def result(mode, speed=None, history=None, followup=None):
    return {"requirements": {"quality_mode": mode}, "context": {}, "speed": speed, "history": history,
            "followup": followup}


class ForkTests(unittest.TestCase):
    def test_near_tie_decisive_outside_and_unscored(self):
        engine = Scorer([(7, -0.6), (8, -1.1), (9, -3.5)])
        judge = forks.ForkJudge(engine, PROTOCOL)
        self.assertIsNone(judge.judge([1, 2], [5, 7, 4], [5, 7, 4]))
        tie = judge.judge([1, 2], [5, 7, 4], [5, 8, 4])
        self.assertEqual((tie["position"], tie["tokens"], tie["near_tie"]), (1, [7, 8], True))
        self.assertAlmostEqual(tie["gap"], 0.5)
        self.assertEqual(engine.calls[-1], ([1, 2], [5, 7], 16, False))  # the shared prefix plus the first token
        self.assertFalse(judge.judge([1], [5, 7], [5, 9])["near_tie"])   # 2.9 nats apart
        outside = judge.judge([1], [5, 7], [5, 3])
        self.assertFalse(outside["near_tie"])
        self.assertIn("outside", outside["reason"])
        self.assertFalse(judge.judge([1], [5, 7], [5, 7, 4])["near_tie"])  # one reply ends where the other goes on
        self.assertIsNone(forks.ForkJudge(object(), PROTOCOL).judge([1], [5, 7], [5, 8])["near_tie"])

    def test_judge_forks_fills_speed_history_and_followup(self):
        engine = Scorer([(7, -0.6), (8, -1.1)])
        prepared = {"speed": [{"id": "chat", "prompt_ids": [1, 2]}], "followup": [{"turn": 2, "prompt_ids": [3]}]}
        r = result("speed",
                   speed={"cases": {"chat": {"status": "measured", "repeatable": False, "variants": [[7, 1], [8, 1]]}}},
                   history={"cases": {"chat": {"identical": False, "first_difference": 0, "fresh": [7], "after": [8]}}},
                   followup={"again": {"turns": {"2": {"identical": False, "first_difference": 0, "first": [8],
                                                       "again": [7]}}}})
        logged = []
        forks.judge_forks(r, engine, prepared, PROTOCOL, logged.append)
        self.assertTrue(forks.near_ties(r["speed"]["cases"]["chat"]["forks"]))
        self.assertTrue(r["history"]["cases"]["chat"]["fork"]["near_tie"])
        self.assertTrue(r["followup"]["again"]["turns"]["2"]["fork"]["near_tie"])
        self.assertEqual(len(engine.calls), 3)
        forks.judge_forks(r, engine, prepared, PROTOCOL, logged.append)
        self.assertEqual(len(engine.calls), 3)  # judged once, also when a resumed round runs it again
        self.assertTrue(all("near-tie" in line for line in logged))

    def test_gates_pass_near_ties_in_speed_only(self):
        tie = {"position": 3, "tokens": [7, 8], "gap": 0.5, "near_tie": True}
        decisive = {"position": 3, "tokens": [7, 9], "gap": 2.9, "near_tie": False}

        def gates(mode, fork):
            speed = {"decode_text": {"source": "generated", "reason": "a fixture"}, "cases": {"chat": {
                "status": "measured", "repeatable": False, "outputs": 2, "forks": [fork], "stalls": None,
                "tool_calls": None}}}
            history = {"cases": {"chat": {"identical": False, "first_difference": 3, "fork": fork}}}
            followup = {"status": "measured", "turns": [{"turn": 2, "cached_tokens": 100}],
                        "again": {"turns": {"2": {"identical": False, "first_difference": 3, "fork": fork}}}}
            found = {g["name"]: g for g in status._gates(result(mode, speed, history, followup))}
            return {name: found[name]["state"] for name in ("repeatable", "request_history", "prompt_cache")}, found

        states, found = gates("speed", tie)
        self.assertEqual(states, {"repeatable": "pass", "request_history": "pass", "prompt_cache": "pass"})
        self.assertIn("near-tie at token 3", found["request_history"]["detail"])
        self.assertEqual(set(gates("speed", decisive)[0].values()), {"fail"})
        self.assertEqual(set(gates("preserve", tie)[0].values()), {"fail"})
        self.assertEqual(set(gates("speed", None)[0].values()), {"fail"})  # an unjudged difference fails


if __name__ == "__main__":
    unittest.main()
