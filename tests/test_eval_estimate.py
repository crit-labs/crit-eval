"""Eval's run-time estimate (crit_eval.run.estimate): every request is known from the prepared data and the protocol, and
the engine's first requests give its prefill and decode costs."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import EVAL_DIR, load_json  # noqa: E402
from crit_eval.run.deadline import RequestDeadline  # noqa: E402
from crit_eval.run.estimate import estimate, fit, PREFILL_TERMS, rates, requests, summary  # noqa: E402

PROTOCOL = {
    "speed": {"warmup_runs": 1, "repetitions": 2},
    "sampled_speed": {"seeds": {"warmup": 0, "repetitions": [1, 2, 3]}},
    "heldout": {"repetitions": 2},
    "followup": {"output_tokens": 8},
    "quality": {"honesty": {"sets": ["conversations"], "reply_tokens": 16}},
    "context": {"curve_from": 2048, "output_tokens": 32, "repetitions": 1, "lower_fraction": 0.25, "granularity": 256,
                "retrieval": {"points": [1024, 2048], "output_tokens": 8}},
}
PREPARED = {
    "speed": [{"id": f"s{n}", "kind": "chat", "prompt_ids": [0] * n, "output_tokens": 32} for n in (100, 1000, 4000)],
    "sampling": {"temperature": 1.0},
    "heldout": [{"id": f"h{n}", "prompt_ids": [0] * n, "output_tokens": 32} for n in (50, 60)],
    "quality": [{"id": "a", "set": "conversations", "prompt_ids": [0] * 20, "continuation_ids": [0] * 16},
                {"id": "b", "set": "conversations", "prompt_ids": [0] * 20, "continuation_ids": [0] * 16},
                {"id": "a@64", "set": "long-64", "prompt_ids": [0] * 64, "continuation_ids": [0] * 16},
                {"id": "b@64", "set": "long-64", "prompt_ids": [0] * 64, "continuation_ids": [0] * 16}],
    "followup": [{"turn": t, "prompt_ids": [0] * n} for t, n in ((1, 500), (2, 520), (3, 540))],
    "retrieval": {"head_ids": [0] * 4, "tail_ids": [0] * 3, "curve_tail_ids": [0] * 4, "filler_ids": [0] * 100,
                  "facts": [{"ids": [0, 0]}] * 3},
}
A, C, P, Q = 1e-4, 1e-8, 0.01, 1e-6   # a synthetic engine: prefill a*n + c*n^2 s, decode p + q*context s per token
INFO = {"max_context": 8192, "speculative": True, "plain": True, "served": True}
ROW = 1e-3


def request(n, m, per_pass=1.0):
    return A * n + C * n * n + (m - 1) * (P + Q * (n + m)) * per_pass


def observed(lengths=(100, 1000, 4000), m=32):
    return {"prefills": [(n, A * n + C * n * n) for n in lengths],
            "decodes": [(n + m, P + Q * (n + m)) for n in lengths]}


def run(observations=None, parts=("speed", "quality", "exactness", "context", "heldout", "history", "followup"),
        prepared=PREPARED, info=INFO, required=8192, **options):
    return estimate(prepared, PROTOCOL, parts, required, observed() if observations is None else observations,
                    info=info, row_seconds=ROW, **options)


class FitTests(unittest.TestCase):
    def test_a_synthetic_engine_gives_back_its_costs(self):
        found = rates(observed()).record()
        self.assertAlmostEqual(found["prefill"]["fixed_seconds"], 0.0, places=9)
        self.assertAlmostEqual(found["prefill"]["seconds_per_token"] / A, 1.0, places=6)
        self.assertAlmostEqual(found["prefill"]["seconds_per_token_squared"] / C, 1.0, places=6)
        self.assertAlmostEqual(found["decode"]["seconds_per_token"] / P, 1.0, places=6)
        self.assertAlmostEqual(found["decode"]["seconds_per_token_per_context_token"] / Q, 1.0, places=6)

    def test_a_fixed_cost_per_request_is_fitted_too(self):
        points = [(n, 0.05 + A * n + C * n * n) for n in (100, 1000, 4000, 16000)]
        t0, a, c = fit(points, PREFILL_TERMS)
        self.assertAlmostEqual(t0, 0.05, places=6)
        self.assertAlmostEqual(a / A, 1.0, places=5)
        self.assertAlmostEqual(c / C, 1.0, places=5)

    def test_one_prompt_length_gives_a_rate_per_token_alone(self):
        t0, a, c = fit([(87, 0.077)] * 3, PREFILL_TERMS)
        self.assertEqual((t0, c), (0.0, 0.0))
        self.assertAlmostEqual(a, 0.077 / 87)

    def test_no_coefficient_goes_negative(self):
        # Faster per token at the longer prompt: a fixed cost explains it, a negative attention term may not.
        t0, a, c = fit([(87, 0.077), (14336, 4.07)], PREFILL_TERMS)
        self.assertEqual(c, 0.0)
        self.assertAlmostEqual(t0 + a * 87, 0.077)
        self.assertAlmostEqual(t0 + a * 14336, 4.07)

    def test_without_a_prefill_and_a_decode_there_is_no_estimate(self):
        self.assertIsNone(run({"prefills": [], "decodes": []}))
        self.assertIsNone(run({"prefills": [(100, 0.1)], "decodes": []}))
        self.assertEqual(summary(None), "estimate: unknown (no request measured yet)")

    def test_a_request_deadline_serves_as_the_observations(self):
        deadline = RequestDeadline({"floor_seconds": 600, "slack": 4})
        for n in (100, 1000, 4000):
            deadline.observe(n, A * n + C * n * n, 33, 32 * (P + Q * (n + 33)))
        self.assertAlmostEqual(run(deadline)["parts"]["speed"], run(observed(m=33))["parts"]["speed"])


class EstimateTests(unittest.TestCase):
    def test_every_part_takes_its_requests_at_the_engines_rates(self):
        found = run(load_seconds=10.0, tokens_per_round=2.0)
        cases = (100, 1000, 4000)
        conversations, long = [(20, 16), (20, 16)], [(64, 16), (64, 16)]
        honesty = [(20, 16), (20, 16)]   # the conversations' replies; the long reply is the curve's first point's
        expected = {
            "lower_context": 10.0 + 3 * sum(request(n, 32) for n in cases),   # 4,096 tokens hold every speed case
            "load": 10.0,
            "history": request(100, 32) + request(4000, 32),
            "followup": sum(request(n, 8) for n in (500, 520, 540)),
            "speed": 3 * sum(request(n, 32) for n in cases),
            "sampled_speed": 4 * sum(request(n, 32) for n in cases),
            "sampling_check": sum(request(n, 32) for n in cases),
            "heldout": 2 * (request(50, 32) + request(60, 32)),
            "quality": sum(request(n, m) + (n + m - 1) * ROW for n, m in conversations + long)
            + sum(2 * request(n, m) + m * ROW for n, m in honesty)
            + request(2016, 32) + 32 * ROW,   # the curve's first reply scored once more
            "context": sum(request(point - 32, 32) for point in (2048, 4096, 8192)),
            "retrieval": request(1016, 8),    # 2,048 is a curve point: its reply is judged instead
            "exactness": sum(request(n, 32, per_pass=2.0) for n in cases),
            "followup_again": sum(request(n, 8) for n in (500, 520, 540)),
            "history_again": request(100, 32) + request(4000, 32),
        }
        self.assertEqual(list(found["parts"]), list(expected))
        for name, seconds in expected.items():
            self.assertAlmostEqual(found["parts"][name], seconds, places=6, msg=name)
        self.assertAlmostEqual(found["total_seconds"], sum(expected.values()), places=6)
        self.assertEqual(found["requests"]["speed"], 9)
        self.assertEqual(found["requests"]["quality"], 4 + 2 * 2 + 1)

    def test_a_quadratic_prefill_makes_the_context_part_dominate_at_262k(self):
        prepared = {**PREPARED, "speed": PREPARED["speed"] + [{"id": "long", "kind": "agent", "prompt_ids": [0] * 14336,
                                                                "output_tokens": 256}]}
        protocol = {**PROTOCOL, "context": {**PROTOCOL["context"], "curve_from": 32768,
                                            "retrieval": {"points": [4096, 32768, 131072], "output_tokens": 64}}}
        info = {**INFO, "max_context": 262144}

        def seconds(c):
            lengths = (100, 1000, 4000, 14336)
            seen = {"prefills": [(n, 0.05 + 2.5e-4 * n + c * n * n) for n in lengths],
                    "decodes": [(n + 256, 0.005) for n in lengths]}
            return estimate(prepared, protocol, ("speed", "quality", "context", "heldout", "history", "followup"),
                            262144, seen, info=info)["parts"]
        quadratic, linear = seconds(2e-9), seconds(0.0)
        self.assertEqual(max(quadratic, key=quadratic.get), "context")
        self.assertGreater(quadratic["context"], 0.5 * sum(quadratic.values()))
        # 262,144 tokens alone: 0.05 + 2.5e-4 * n + 2e-9 * n^2 of prefill.
        self.assertGreater(quadratic["context"], 0.05 + 2.5e-4 * 261888 + 2e-9 * 261888 ** 2)
        self.assertGreater(quadratic["context"], 2 * linear["context"])

    def test_parts_that_will_not_run_are_left_out(self):
        self.assertEqual(list(run(parts=("speed",))["parts"]), ["speed", "sampled_speed", "sampling_check"])
        # Timing parts by their own names, and parts the caller knows will not run.
        self.assertEqual(list(run(parts=("speed", "heldout"), skip=("sampled_speed",))["parts"]), ["speed", "heldout"])
        unsampled = {k: v for k, v in PREPARED.items() if k not in ("sampling", "followup", "heldout")}
        self.assertEqual(list(run(parts=("speed", "followup", "heldout"), prepared=unsampled)["parts"]), ["speed"])
        # Without a required context: no lower context, curve or retrieval.
        self.assertNotIn("context", run(required=None, info={**INFO, "max_context": 8192})["parts"])
        self.assertNotIn("lower_context", run(required=None)["parts"])
        # A profile that loads the engine once.
        self.assertNotIn("lower_context", run(settings={**PROTOCOL["context"], "measure_lower": False})["parts"])
        # A plain engine checks no exactness; nor does a served speculative one that cannot decode plainly.
        self.assertNotIn("exactness", run(info={**INFO, "speculative": False})["parts"])
        self.assertNotIn("exactness", run(info={**INFO, "plain": False})["parts"])
        # The follow-up turns skip when the last does not fit; quality sets beyond the context too.
        short = run(info={**INFO, "max_context": 70})
        self.assertNotIn("followup", short["parts"])
        self.assertEqual(short["requests"]["quality"], 2 + 2 * 2)
        # The load is counted only when known.
        self.assertNotIn("load", run()["parts"])

    def test_a_second_engine_costs_its_load(self):
        separate = run(info={**INFO, "plain": False, "served": False}, load_seconds=7.0)
        shared = run(load_seconds=7.0)
        self.assertAlmostEqual(separate["parts"]["exactness"] - shared["parts"]["exactness"], 7.0)

    def test_parts_that_ran_count_as_they_took(self):
        found = run(measured={"lower_context": 40.0, "load": 12.0, "history": 3.0})
        self.assertEqual([found["parts"][k] for k in ("lower_context", "load", "history")], [40.0, 12.0, 3.0])
        self.assertNotIn("history", found["requests"])
        self.assertAlmostEqual(found["total_seconds"], sum(found["parts"].values()))

    def test_a_pinned_continuation_adds_two_free_generations_per_case(self):
        plain = requests(PREPARED, PROTOCOL, ("speed",), 8192)
        pinned = requests(PREPARED, PROTOCOL, ("speed",), 8192, pinned=True)
        self.assertEqual((len(plain["speed"]), len(pinned["speed"])), (9, 15))
        self.assertEqual(len(plain["sampled_speed"]), 12)

    def test_the_summary_names_each_part(self):
        found = {"parts": {"speed": 120.0, "quality": 30.0, "context": 690.0}, "total_seconds": 840.0}
        self.assertEqual(summary(found), "estimate: about 14 min (speed 2 min, quality 30 s, context 12 min)")

    def test_the_protocol_holds_what_the_estimate_reads(self):
        protocol = load_json(EVAL_DIR / "protocol.json")
        found = estimate(PREPARED, protocol, ("speed", "quality", "exactness", "context", "heldout", "history",
                                              "followup"), 8192, observed(), info=INFO)
        self.assertIn("speed", found["parts"])
        self.assertTrue(summary(found).startswith("estimate: about "))


if __name__ == "__main__":
    unittest.main()
