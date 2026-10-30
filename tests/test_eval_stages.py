"""crit eval run's stages (protocol stages): stage 1 on every engine, the budget choosing how far a run goes, and
the certification serving the required context (mock adapter, no GPU)."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crit_eval import EVAL_DIR, load_json  # noqa: E402
from crit_eval.limits import context  # noqa: E402
from crit_eval.reports import status  # noqa: E402
from crit_eval.run import quality_checks, runner, stages  # noqa: E402
from crit_eval.run.estimate import Rates  # noqa: E402
from test_eval import FACTS, MOCK, fixture  # noqa: E402

PROTOCOL = load_json(EVAL_DIR / "protocol.json")
REQUIRED = 65536


def run(directory, **options):
    data = Path(directory)
    fixture(data)
    # The mock answers the retrieval facts it was given.
    return runner.run(MOCK, "unused-model", data, data / "result.json", {"facts_from": FACTS}, log=lambda message: None,
                      context_tokens=REQUIRED, **options)


def gate(result, name):
    return next(g for g in result["status"]["gates"] if g["name"] == name)


class StagedRunTests(unittest.TestCase):
    def test_a_run_with_room_in_its_budget_reaches_stage_3_without_the_required_context(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory)
        self.assertEqual(result["stages"]["reached"], 3)
        self.assertFalse(result["stages"]["certify"])
        # The stage 1 request is the warmup of each short case's two repetitions (the fixture has no long prompt); a
        # third follows when the two disagree (protocol speed.extra_repetition), as the mock's sub-ms timings may.
        for case in result["speed"]["cases"].values():
            self.assertEqual([s["warmup"] for s in case["samples"]][:3], [True, False, False])
            self.assertLessEqual(len(case["samples"]), 4)
        self.assertEqual(result["sampled_speed"]["status"], "measured")
        self.assertEqual(result["sampled_speed"]["repetitions"], 2)
        # Exactness on the two shortest cases, on the shared engine.
        self.assertEqual(sorted(result["exactness"]["cases"]), ["long", "short"])
        # The curve stops at its first point: no request fills the required context, so capacity is unmeasured.
        self.assertEqual(list(result["context"]["points"]), ["32768"])
        self.assertFalse(result["context"]["certified"])
        self.assertEqual(result["context"]["retrieval"]["points"][str(REQUIRED)]["status"], "skipped")
        self.assertEqual(gate(result, "capacity")["state"], "unmeasured")
        self.assertIn("--certify", gate(result, "capacity")["detail"])
        self.assertEqual(gate(result, "retrieval")["state"], "pass")
        # The lower context ran once, after the engine at the requirement closed.
        self.assertIn("lower_context", result["timing"]["parts"])
        self.assertEqual(list(result["timing"]["parts"])[-1], "lower_context")
        self.assertEqual(result["context"]["lower"]["context"], 16384)
        self.assertEqual(sorted(result["context"]["lower"]["cases"]), ["long", "short"])
        # The request history repeats the shortest case alone.
        self.assertEqual(list(result["history"]["cases"]), ["short"])
        # The held-out prompts need a baseline that measured them.
        self.assertEqual(result["heldout"]["status"], "skipped")
        # What stage 1 gives without a long prompt: decode projected to the requirement and the TTFT floor.
        creep = result["context"]["creep"]
        self.assertEqual(creep["measured_up_to"], 32768)
        self.assertTrue(creep["extrapolated"])
        self.assertIsNotNone(creep["decode_tps_at_requirement"])
        self.assertEqual(result["speed"]["ttft_floor_case"], "short")
        # The first estimate came right after the first speed case.
        self.assertEqual(sorted(result["stages"]["first_estimate"]), ["1", "2", "3", "certify"])

    def test_a_spent_budget_stops_after_stage_1_and_says_what_it_left_out(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, budget_seconds=1e-6)
        self.assertEqual(result["stages"]["reached"], 1)
        # After the first case, stage 1 leaves out what no longer fits: coverage is unmeasured, not failed.
        cases = result["speed"]["cases"]
        self.assertEqual(len(cases["short"]["samples"]), 1)
        self.assertEqual((cases["long"]["status"], cases["long"]["over_budget"]), ("skipped", True))
        self.assertIn("over budget", cases["long"]["reason"])
        self.assertEqual(gate(result, "coverage")["state"], "unmeasured")
        for part in ("followup", "sampled_speed", "heldout"):
            self.assertEqual(result[part]["status"], "skipped")
            self.assertIn("stopped after stage 1", result[part]["reason"])
        self.assertEqual(result["exactness"]["status"], "unmeasured")
        self.assertNotIn("points", result["context"])
        self.assertIsNone(result["context"]["lower"]["context"])
        self.assertNotIn("lower_context", result["timing"]["parts"])
        self.assertEqual(gate(result, "capacity")["state"], "unmeasured")
        self.assertEqual(gate(result, "request_history")["state"], "pass")
        self.assertIn("history_again", result["timing"]["parts"])

    def test_a_hung_engine_keeps_what_the_run_measured(self):
        from unittest.mock import patch
        from crit_eval.run.deadline import Hung
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_measure",
                                                                      side_effect=Hung("no answer within 893 s")):
            result = run(directory)
            stored = load_json(Path(directory) / "result.json")
        self.assertEqual(result["aborted"]["reason"], "hung")
        self.assertEqual(stored["aborted"]["detail"], "no answer within 893 s")

    def test_a_stage_cap_stops_a_run_there(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage=2)
        self.assertEqual(result["stages"]["reached"], 2)
        self.assertEqual(list(result["context"]["points"]), ["32768"])
        self.assertEqual(result["sampled_speed"]["status"], "skipped")
        self.assertNotIn("again", result["followup"])

    def test_a_certified_run_serves_the_required_context_and_passes_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, certify=True, budget_seconds=1e-6)
        self.assertEqual(result["stages"]["reached"], 3)
        self.assertIsNone(result["stages"]["budget_seconds"])
        self.assertEqual(list(result["context"]["points"]), ["32768", str(REQUIRED)])
        self.assertTrue(result["context"]["certified"])
        self.assertEqual(gate(result, "capacity")["state"], "pass")
        self.assertTrue(result["context"]["retrieval"]["points"][str(REQUIRED)]["passed"])

    def test_stage_full_runs_the_parts_as_before_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage="full", warmups=0, repetitions=2)
        self.assertIsNone(result["stages"])
        self.assertEqual(list(result["context"]["points"]), ["32768", str(REQUIRED)])
        self.assertEqual(gate(result, "capacity")["state"], "pass")

    def test_certify_is_for_a_staged_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                run(directory, stage="full", certify=True)


class StagePlanTests(unittest.TestCase):
    PREPARED = {"speed": [{"prompt_ids": [0] * 100, "output_tokens": 256}, {"prompt_ids": [0] * 14336, "output_tokens": 256}],
                "quality": [{"set": "conversations", "prompt_ids": [0] * 30, "continuation_ids": [0] * 256},
                            {"set": "long-32768", "prompt_ids": [0] * 32768, "continuation_ids": [0] * 256}],
                "sampling": {"temperature": 1.0},
                "followup": [{"prompt_ids": [0] * 8192}, {"prompt_ids": [0] * 8300}],
                "retrieval": {"facts": []}}
    INFO = {"max_context": 200000, "speculative": True}

    def plan(self, certify=False):
        settings = stages.staged_settings(PROTOCOL["context"], PROTOCOL["stages"], certify)
        return stages.stage_requests(self.PREPARED, PROTOCOL, 200000, self.INFO, settings, certify=certify)

    def test_each_stage_adds_its_own_requests(self):
        planned = self.plan()
        self.assertEqual(len(planned[1]["speed"]), 2)
        self.assertEqual([r[0] for r in planned[1]["quality"]], [30, 30, 30])   # the text, the reply, its score
        self.assertEqual(len(planned[2]["speed"]), 2)    # the short case twice more, the long one not again
        self.assertEqual(planned[2]["context"], [(32768 - 256, 256, 0, False)])
        self.assertEqual(planned[2]["retrieval"][0][0] + planned[2]["retrieval"][0][1], 4096)
        self.assertEqual(len(planned[3]["sampled_speed"]), 6)   # a warmup and two seeds per case
        self.assertEqual(planned["certify"], {})
        self.assertEqual(self.plan(certify=True)["certify"]["context"], [(200000 - 256, 256, 0, False)])

    def test_the_budget_picks_the_highest_stage_that_fits(self):
        planned = self.plan()
        slow = Rates((2.5, 1 / 55, 0.0), (0.1, 0.0))      # 55 tok/s prefill, 10 tok/s decode
        fast = Rates((0.1, 1 / 3000, 0.0), (1 / 60, 0.0))
        self.assertEqual(stages.choose(planned, slow, 600, 2700)[0], 1)
        stage, costs = stages.choose(planned, fast, 600, 2700)
        self.assertEqual(stage, 3)
        self.assertLess(costs[2] + costs[3], 2100)
        self.assertEqual(stages.choose(planned, None, 0, 2700), (1, {}))

    def test_the_curve_without_certification_stops_at_its_first_point(self):
        settings = {"curve_from": 32768, "curve_points": "first"}
        self.assertEqual(context.curve_points(200000, settings), [32768])
        self.assertEqual(context.curve_points(200000, {**settings, "curve_points": "ends"}), [32768, 200000])
        self.assertEqual(context.curve_points(16384, settings), [16384])

    def test_decode_creep_is_fitted_across_contexts_and_projected(self):
        def row(prompt, seconds_per_token):
            return {"status": "measured", "prompt_tokens": prompt,
                    "samples": [{"output_tokens": 101, "ttft_ms": 0.0, "request_ms": seconds_per_token * 100 * 1000}]}
        # 10 ms per token plus 1 ms per 10,000 context tokens.
        result = {"speed": {"cases": {"a": row(899, 0.01009), "b": row(9899, 0.01099)}},
                  "context": {"points": {"32768": row(32667, 0.013276)}}}
        found = stages.creep(result, 200000)
        self.assertAlmostEqual(found["decode_tps_at_requirement"], 1 / 0.03, places=0)
        self.assertAlmostEqual(found["slowdown_at_requirement"], 1 - (1 / 0.03) / (1 / 0.01009), places=2)
        self.assertIsNone(stages.creep({"speed": {"cases": {"a": row(899, 0.01)}}}, 200000))

    def test_the_memory_fit_compares_free_memory_with_the_kv_cache_at_the_requirement(self):
        found = stages.memory_fit({"free_bytes": 4e9}, {"kv_bytes_per_token": 24576}, None, 200000)
        self.assertEqual(found["kv_bytes_at_requirement"], 24576 * 200000)
        self.assertFalse(found["fits_if_allocated_late"])
        self.assertEqual(stages.memory_fit({"free_bytes": 4e9}, {}, {"kv_bytes_per_token": 1000}, 200000)
                         ["kv_bytes_per_token_source"], "model")

    def test_a_later_quality_set_joins_the_first_ones(self):
        first = {"status": "scored", "sets": {"a/prompt": {"passed": True}}, "buckets": {"a/prompt": {}},
                 "skipped": {"long": "a stage 2 set"}, "honesty": {"passed": True}, "seconds": 1.0,
                 "timing": {"texts": [], "metrics_seconds": 0.5}, "passed": True}
        more = {"status": "scored", "sets": {"long/prompt": {"passed": False}}, "buckets": {"long/prompt": {}},
                "skipped": {}, "honesty": {"passed": None}, "seconds": 2.0,
                "timing": {"texts": [], "metrics_seconds": 0.25}, "passed": False}
        merged = quality_checks.merge_quality(first, more)
        self.assertEqual(sorted(merged["sets"]), ["a/prompt", "long/prompt"])
        self.assertEqual(merged["skipped"], {})
        self.assertFalse(merged["passed"])
        self.assertEqual(merged["honesty"], {"passed": True})
        self.assertEqual(merged["seconds"], 3.0)


class CapacityGateTests(unittest.TestCase):
    def test_an_uncertified_run_leaves_capacity_unmeasured_with_the_memory_it_needs(self):
        found = status._context_gates({"context": {"tokens": 200000, "certified": False, "points": {},
                                                   "fit": {"free_after_load_bytes": 2e9,
                                                           "kv_bytes_at_requirement": 5e9}}})
        capacity = found[0]
        self.assertEqual(capacity["state"], "unmeasured")
        self.assertIn("2.00 GB free after the load", capacity["detail"])


if __name__ == "__main__":
    unittest.main()
