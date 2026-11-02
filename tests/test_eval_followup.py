"""Eval's follow-up turns (protocol followup): measured with whatever prompt cache the engine has, and its cached path
must answer as its fresh one did."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval.run import chat, estimate, history, runner, stages, timing
from crit_eval.reports import status  # noqa: E402

TURNS = [{"turn": 1, "prompt_ids": list(range(1, 101))},
         {"turn": 2, "prompt_ids": list(range(1, 121))},
         {"turn": 3, "prompt_ids": list(range(1, 141))}]


class Engine(chat.ChatEngine):
    """An engine behind a server: with `cache`, it serves the longest prompt prefix it saw before from its cache, with
    `exact` only a prompt it saw whole; with `drifts`, a reply after a cache hit differs from a fresh one."""

    def __init__(self, cache=False, drifts=False, exact=False):
        self.cache, self.drifts, self.exact, self.seen, self.asked = cache or exact, drifts, exact, [], []

    def generate(self, prompt, max_new_tokens, on_tokens, prompt_cache=False):
        self.asked.append(prompt_cache)
        cached = max((len(p) for p in self.seen if prompt[:len(p)] == p and (not self.exact or p == prompt)),
                     default=0) if self.cache else 0
        self.seen.append(list(prompt))
        tokens = [len(prompt) % 50 + (1 if self.drifts and cached else 0)] * max_new_tokens
        on_tokens(tokens)
        return {"tokens": tokens, "served": {"prompt_tokens": len(prompt), "output_tokens": max_new_tokens,
                                             "cached_tokens": cached, "finish_reason": "length"}}


class FollowupTests(unittest.TestCase):
    protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")

    def measure(self, engine, max_context=1000, **info):
        prepared, info = {"followup": TURNS}, {"max_context": max_context, **info}
        first = history.run_followup(engine, info, prepared, self.protocol, lambda m: None)
        if first["status"] == "measured":
            first["again"] = history.compare_followup(
                first, history.run_followup(engine, info, prepared, self.protocol, lambda m: None))
        gate = next(g for g in status._gates({"requirements": {"exactness": "reported"}, "context": {"tokens": None,
                    "reason": "-", "missing": []}, "followup": first}) if g["name"] == "prompt_cache")
        return first, gate

    def test_without_a_cache_every_turn_prefills_and_the_gate_passes(self):
        engine = Engine()
        first, gate = self.measure(engine)
        self.assertEqual([t["cached_tokens"] for t in first["turns"]], [0, 0, 0])
        self.assertEqual(gate["state"], "pass")
        self.assertTrue(all(engine.asked))   # every turn allows a prompt cache
        # The KPIs cover the turns after the first: the prefill rate counts every prompt token.
        self.assertAlmostEqual(first["turns"][1]["prefill_tps"], 120 * 1000 / first["turns"][1]["ttft_ms"])

    def test_a_cache_serves_the_earlier_turns_and_its_answers_hold(self):
        first, gate = self.measure(Engine(cache=True))
        self.assertEqual([t["cached_tokens"] for t in first["turns"]], [0, 100, 120])
        # The prefill rate counts only the tokens the engine computed.
        self.assertAlmostEqual(first["turns"][2]["prefill_tps"], 20 * 1000 / first["turns"][2]["ttft_ms"])
        self.assertEqual(gate["state"], "pass")
        self.assertIn("cached prompt tokens 0, 100, 120", gate["detail"])

    def test_a_cache_that_changes_the_answer_fails(self):
        first, gate = self.measure(Engine(cache=True, drifts=True))
        self.assertEqual(gate["state"], "fail")
        self.assertIn("turn 1 differs at token 0", gate["detail"])   # cached the second time, fresh the first

    def test_an_exact_prompt_cache_hits_only_in_the_second_pass(self):
        # Cached tokens of 0 in the first pass declare no "no prompt cache": this cache serves each turn again whole.
        first, gate = self.measure(Engine(exact=True, drifts=True))
        self.assertEqual([t["cached_tokens"] for t in first["turns"]], [0, 0, 0])
        self.assertEqual(gate["state"], "fail")

    def test_an_engine_that_declares_no_prompt_cache_is_not_sent_the_turns_and_passes(self):
        engine = Engine()
        first, gate = self.measure(engine, prompt_cache=False)
        self.assertEqual((first["status"], first["no_prompt_cache"], engine.seen), ("skipped", True, []))
        self.assertEqual(gate["state"], "pass")
        self.assertIn("declares no prompt cache", gate["detail"])
        # Neither pass is priced: not in the run's estimate, nor in stage 2 and 3 of a staged run.
        protocol, prepared = self.protocol, {"speed": [], "followup": TURNS}
        for declared, expected in ((None, {"followup", "followup_again"}), (False, set())):
            info = {"max_context": 1000, **({} if declared is None else {"prompt_cache": declared})}
            found = estimate.requests(prepared, protocol, ("followup",), None, info=info)
            self.assertEqual(set(found), expected)
            planned = stages.stage_requests(prepared, protocol, None, info, protocol["context"])
            self.assertEqual({part for stage in (2, 3) for part in planned[stage]} & {"followup", "followup_again"},
                             expected)
        # crit serve publishes it among the engine facts.
        self.assertIn("prompt_cache", chat.FACTS)

    def test_turns_that_do_not_fit_are_skipped_and_unmeasured(self):
        first, gate = self.measure(Engine(), max_context=150)
        self.assertEqual((first["status"], gate["state"]), ("skipped", "unmeasured"))

    def test_timed_requests_still_refuse_a_prompt_cache(self):
        engine = Engine(cache=True)
        timing.timed_generate(engine, [1, 2, 3], 4)
        timing.timed_generate(engine, [1, 2, 3], 4, prompt_cache=True)
        self.assertEqual(engine.asked, [False, True])


if __name__ == "__main__":
    unittest.main()
