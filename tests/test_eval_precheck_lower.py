"""The gate pre-check's lower-context item: the certified gate's comparison on LOWER_SAMPLES samples of each case at
each context, its tolerance the protocol's or the case's own spread, failed where a case is slower than that,
unverified where so few samples spread too wide to judge."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crit_eval.run import precheck  # noqa: E402

SETTINGS = {"tolerance": 0.05}


def case(decode, ttft, engine_first_token=None):
    """A measured case of per-sample decode tok/s and first-token ms (engine's own first token where given)."""
    samples = [{"decode_tps": d, "ttft_ms": t, **({"first_token_ms": e} if e is not None else {})}
               for d, t, e in zip(decode, ttft, engine_first_token or [None] * len(decode))]
    middle = lambda values: sorted(values)[len(values) // 2]  # noqa: E731
    return {"status": "measured", "samples": samples, "decode_tps": middle(decode), "ttft_ms": middle(ttft)}


class LowerContextTests(unittest.TestCase):
    def test_a_case_as_fast_at_the_lower_context_passes(self):
        high = {"cases": {"short": case([100, 101, 100], [30, 30, 31])}}
        low = {"cases": {"short": case([100, 100, 99], [30, 31, 30])}}
        item = precheck.lower_item(high, low, 65536, SETTINGS)
        self.assertEqual(item["state"], "passed", item)

    def test_one_slow_sample_among_three_does_not_fail_a_case(self):
        # The single-sample pre-check failed on whichever case one slow request hit (35 -> 83 ms on one run):
        # the median of three is the case's time, and the slow sample's spread widens its tolerance.
        high = {"cases": {"short": case([100, 100, 100], [35, 35, 36])}}
        low = {"cases": {"short": case([100, 100, 100], [35, 83, 36])}}
        self.assertEqual(precheck.lower_item(high, low, 65536, SETTINGS)["state"], "passed")

    def test_a_case_slower_on_every_sample_fails(self):
        high = {"cases": {"long": case([50, 50, 51], [800, 805, 802])}}
        low = {"cases": {"long": case([50, 50, 50], [960, 965, 962])}}
        item = precheck.lower_item(high, low, 65536, SETTINGS)
        self.assertEqual(item["state"], "failed", item)
        self.assertIn("long first token", item["detail"])

    def test_samples_too_spread_to_judge_are_unverified_not_failed(self):
        high = {"cases": {"long": case([50, 50, 50], [800, 1100, 820])}}
        low = {"cases": {"long": case([50, 50, 50], [1300, 1310, 1290])}}
        item = precheck.lower_item(high, low, 65536, SETTINGS)
        self.assertEqual(item["state"], "unverified", item)
        self.assertIn("spread too wide", item["detail"])

    def test_the_engines_own_first_token_is_judged_where_both_contexts_report_it(self):
        # A reopened host's first requests pay its cold caches outside the engine: the engine's own time decides.
        high = {"cases": {"long": case([50] * 3, [800, 801, 802], [700, 700, 701])}}
        low = {"cases": {"long": case([50] * 3, [900, 901, 902], [701, 700, 702])}}
        self.assertEqual(precheck.lower_item(high, low, 65536, SETTINGS)["state"], "passed")

    def test_speed_samples_warm_each_case_and_time_it_lower_samples_times(self):
        calls = []

        def timed(engine, prompt, output_tokens):
            calls.append(len(prompt))
            return None, {"decode_tps": 10.0 + len(calls), "ttft_ms": 5.0}
        original = precheck.timed_generate
        precheck.timed_generate = timed
        try:
            found = precheck.speed_samples(None, [{"id": "a", "prompt_ids": [1, 2], "output_tokens": 4}])
        finally:
            precheck.timed_generate = original
        self.assertEqual(len(calls), 1 + precheck.LOWER_SAMPLES)
        self.assertEqual(len(found["cases"]["a"]["samples"]), precheck.LOWER_SAMPLES)
        self.assertEqual(found["cases"]["a"]["decode_tps"], 13.0)   # the median of 12, 13, 14


if __name__ == "__main__":
    unittest.main()
