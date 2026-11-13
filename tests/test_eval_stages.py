"""crit eval run's stages (protocol stages): stage 1 on every engine, the budget choosing how far a run goes, and
the certification serving the required context (mock adapter, no GPU)."""
import json
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


def run(directory, baseline=None, mock=None, **options):
    """A run of the mock engine on the fixture; mock: more of the mock's configuration (its test knobs)."""
    data = Path(directory)
    prepared = fixture(data)
    prepared["heldout"] = [{"id": "zh", "language": "Chinese", "kind": "prose", "prompt_ids": [7, 8, 9],
                            "output_tokens": 12}]
    (data / "prepared.json").write_text(json.dumps(prepared))
    # The mock answers the retrieval facts it was given.
    return runner.run(MOCK, "unused-model", data, data / "result.json", {"facts_from": FACTS, **(mock or {})}, log=lambda message: None,
                      context_tokens=REQUIRED, baseline=baseline, **options)


def gate(result, name):
    return next(g for g in result["status"]["gates"] if g["name"] == name)


class StagedRunTests(unittest.TestCase):
    def test_a_run_with_room_in_its_budget_reaches_stage_3_without_the_required_context(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory)
        self.assertEqual(result["stages"]["reached"], 3)
        self.assertFalse(result["stages"]["certify"])
        # The stage 1 request is the warmup of each short case's repetition (the fixture has no long prompt); a second
        # and third follow when they disagree (protocol speed.extra_repetition), as the mock's sub-ms timings may.
        for case in result["speed"]["cases"].values():
            self.assertEqual([s["warmup"] for s in case["samples"]][:2], [True, False])
            self.assertLessEqual(len(case["samples"]), 4)
        self.assertEqual(result["sampled_speed"]["status"], "measured")
        self.assertEqual(result["sampled_speed"]["repetitions"], 1)
        # Exactness on the two shortest cases, on the shared engine.
        self.assertEqual(sorted(result["exactness"]["cases"]), ["long", "short"])
        # The curve stops at its first point: no request fills the required context, so capacity is unmeasured.
        self.assertEqual(list(result["context"]["points"]), ["32768"])
        self.assertFalse(result["context"]["certified"])
        self.assertEqual(result["context"]["retrieval"]["points"][str(REQUIRED)]["status"], "skipped")
        self.assertEqual(gate(result, "capacity")["state"], "unmeasured")
        self.assertIn("--certify", gate(result, "capacity")["detail"])
        self.assertEqual(gate(result, "retrieval")["state"], "pass")
        # The lower context opens a second engine: a certified run's alone.
        self.assertNotIn("lower_context", result["timing"]["parts"])
        self.assertIsNone(result["context"]["lower"]["context"])
        self.assertEqual(gate(result, "lower_context")["state"], "unmeasured")
        self.assertIn("--certify", gate(result, "lower_context")["detail"])
        # The request history repeats the shortest case alone.
        self.assertEqual(list(result["history"]["cases"]), ["short"])
        # The held-out prompts run in stage 1, right after the speed cases, with no baseline needed to measure them.
        self.assertEqual(result["heldout"]["status"], "measured")
        parts = list(result["timing"]["parts"])
        self.assertEqual(parts.index("heldout"), parts.index("speed") + 1)
        self.assertEqual(gate(result, "generality")["state"], "unmeasured")
        # What stage 1 gives without a long prompt: decode projected to the requirement and the TTFT floor.
        creep = result["context"]["creep"]
        # The curve's first point fills its drawn share of 32,768 tokens (protocol jitter).
        self.assertEqual(creep["measured_up_to"], result["context"]["points"]["32768"]["filled_context"])
        self.assertTrue(0.8 * 32768 <= creep["measured_up_to"] <= 32768)
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
        self.assertEqual(result["followup"]["status"], "skipped")
        self.assertIn("stopped after stage 1", result["followup"]["reason"])
        # The sampled suite is stage 1's: every candidate's decode KPI is the sampled one.
        self.assertEqual(result["sampled_speed"]["status"], "measured")
        # Stage 1 measures the held-out prompts whatever the budget left: they are every later Eval's baseline.
        self.assertEqual(result["heldout"]["status"], "measured")
        self.assertIn("heldout", result["history"]["served_between"])
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

    def test_a_run_of_the_context_part_alone_measures_the_curve_and_retrieval(self):
        # Without the speed part there are no rates to budget by: the curve and retrieval still run.
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, parts=("context",))
        self.assertEqual(list(result["context"]["points"]), ["32768"])
        self.assertEqual(result["context"]["retrieval"]["status"], "measured")
        self.assertEqual(gate(result, "retrieval")["state"], "pass")
        self.assertNotIn("speed", result)

    def test_a_stage_cap_stops_a_run_there(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage=2)
        self.assertEqual(result["stages"]["reached"], 2)
        self.assertEqual(list(result["context"]["points"]), ["32768"])
        self.assertEqual(result["sampled_speed"]["status"], "measured")
        self.assertNotIn("again", result["followup"])
        # Stage 2 repeats the short cases: their outputs are compared.
        self.assertEqual(gate(result, "repeatable")["state"], "pass")
        self.assertEqual(gate(result, "repeatable")["detail"], "2/2 cases repeat their output")
        # The follow-up turns run again in stage 3: a stage 2 run says so.
        followup = {"status": "measured", "turns": [{"turn": 1, "cached_tokens": 0}]}
        found = status._prompt_cache_gate({"followup": followup, "stages": {"reached": 2}})
        self.assertEqual(found["state"], "unmeasured")
        self.assertIn("a stage 3 part, and this run stopped after stage 2", found["detail"])

    def test_a_case_decoded_once_repeats_nothing(self):
        # Stage 1 decodes each speed case once: the repeatable gate has no output to compare, and says so instead of
        # counting the cases as repeating.
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage=1)
        repeatable = gate(result, "repeatable")
        self.assertEqual((repeatable["state"], repeatable["detail"]),
                         ("unmeasured", "every speed case decoded once: no output to repeat"))
        self.assertEqual({c["outputs"] for c in result["speed"]["cases"].values()}, {1})
        # A result that compares some cases names the others.
        cases = {"a": {"status": "measured", "repeatable": True, "outputs": 2, "samples": []},
                 "b": {"status": "measured", "repeatable": True, "outputs": 1, "samples": [{"warmup": False}]}}
        found = next(g for g in status._gates({"speed": {"cases": cases}, "context": {}, "requirements": {
            "exactness": "reported"}}) if g["name"] == "repeatable")
        self.assertEqual((found["state"], found["detail"]), ("pass", "1/1 cases repeat their output; decoded once, "
                                                                     "not compared: b"))

    def test_a_certified_run_serves_the_required_context_and_passes_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, certify=True, budget_seconds=1e-6)
        self.assertEqual(result["stages"]["reached"], 3)
        self.assertIsNone(result["stages"]["budget_seconds"])
        self.assertEqual(list(result["context"]["points"]), ["32768", str(REQUIRED)])
        self.assertTrue(result["context"]["certified"])
        self.assertEqual(gate(result, "capacity")["state"], "pass")
        self.assertTrue(result["context"]["retrieval"]["points"][str(REQUIRED)]["passed"])
        # The lower context ran once, after the engine at the requirement closed.
        self.assertEqual(list(result["timing"]["parts"])[-1], "lower_context")
        self.assertEqual(result["context"]["lower"]["context"], 16384)
        self.assertEqual(sorted(result["context"]["lower"]["cases"]), ["long", "short"])
        self.assertNotEqual(gate(result, "lower_context")["state"], "unmeasured")

    def test_stage_full_runs_the_parts_as_before_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage="full", warmups=0, repetitions=2)
        self.assertIsNone(result["stages"])
        self.assertEqual(list(result["context"]["points"]), ["32768", str(REQUIRED)])
        self.assertEqual(gate(result, "capacity")["state"], "pass")
        self.assertEqual(result["context"]["lower"]["context"], 16384)   # without --certify, as before stages

    def test_the_first_certified_eval_judges_generality_against_the_first_staged_one(self):
        # protocol stages 1: a candidate's first staged Eval measured the held-out prompts, so the certification's
        # generality gate has a baseline that did.
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            staged = run(first, stage=1)
            certified = run(second, certify=True, baseline=staged)
        self.assertEqual((staged["stages"]["reached"], staged["heldout"]["status"]), (1, "measured"))
        generality = gate(certified, "generality")
        self.assertNotEqual(generality["state"], "unmeasured", generality["detail"])
        self.assertIn("held-out decode", generality["detail"])

    def test_an_engines_own_expert_cache_counters_are_read_over_each_speed_suite(self):
        # An engine whose requests report no expert reads but whose telemetry() counts its expert cache: the counters
        # over the speed suites' requests are the hit rate its memory tiers price (protocol headroom.tiers).
        from crit_eval.limits import headroom
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            result = runner.run(MOCK, "unused-model", data, data / "result.json",
                                {"facts_from": FACTS, "expert_cache": {"reads_per_token": 10, "hit_rate": 0.75}},
                                log=lambda message: None, context_tokens=REQUIRED)
        self.assertEqual(sorted(result["telemetry"]), ["sampled_speed", "speed"])
        for window in result["telemetry"].values():
            self.assertGreater(window["reads"], 0)
            self.assertAlmostEqual(window["hit_rate"], 0.75, places=2)
        self.assertEqual(headroom.measured_hit_rate(result)[1], "engine telemetry")
        # An engine without counters records none.
        with tempfile.TemporaryDirectory() as directory:
            self.assertNotIn("telemetry", run(directory, stage=1))

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
        self.assertEqual(len(planned[2]["speed"]), 1)    # the short case once more, the long one not again
        self.assertEqual(planned[2]["context"], [(32768 - 256, 256, 0, False)])
        self.assertEqual(planned[2]["retrieval"][0][0] + planned[2]["retrieval"][0][1], 4096)
        self.assertEqual(len(planned[1]["sampled_speed"]), 4)   # a warmup and one seed per case, in stage 1
        self.assertNotIn("sampled_speed", planned[3])
        self.assertNotIn("lower_context", planned[3])
        self.assertEqual(planned["certify"], {})
        # The held-out prompts are stage 1's, priced at their nominal lengths; a run without the part plans none.
        held = {**self.PREPARED, "heldout": [{"prompt_ids": [0] * 200, "output_tokens": 64}]}
        settings = stages.staged_settings(PROTOCOL["context"], PROTOCOL["stages"], False)
        found = stages.stage_requests(held, PROTOCOL, 200000, self.INFO, settings)
        self.assertEqual(found[1]["heldout"], [(200, 64, 0, False)])
        self.assertFalse(any("heldout" in found[stage] for stage in (2, 3, "certify")))
        self.assertNotIn("heldout", stages.stage_requests(held, PROTOCOL, 200000, self.INFO, settings, heldout=False)[1])
        certified = self.plan(certify=True)["certify"]
        self.assertEqual(certified["context"], [(200000 - 256, 256, 0, False)])
        self.assertEqual(len(certified["lower_context"]), 2)   # each speed case once at the lower context

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
        found = stages.memory_fit({"free_bytes": 4e9}, {"kv_bytes_per_token": 24576}, 200000)
        self.assertEqual(found["kv_bytes_at_requirement"], 24576 * 200000)
        self.assertFalse(found["fits_if_allocated_late"])
        self.assertIsNone(found["unmeasured"])
        self.assertTrue(stages.memory_fit({"free_bytes": 4e9}, {"kv_bytes_per_token": 1000}, 200000)
                        ["fits_if_allocated_late"])

    def test_the_memory_fit_takes_no_kv_count_but_the_engines(self):
        # A model's geometry says nothing of the precision the engine stores its KV at: without the engine's count the
        # KV cache at the requirement is unmeasured, never priced at 16 bits.
        for info in ({}, {"kv_bytes_per_token": None}, {"kv_bytes_per_token": 0}):
            with self.subTest(info=info):
                found = stages.memory_fit({"free_bytes": 4e9}, info, 200000)
                self.assertEqual(found, {"free_after_load_bytes": 4e9, "kv_bytes_per_token": None,
                                         "kv_bytes_at_requirement": None, "fits_if_allocated_late": None,
                                         "unmeasured": stages.KV_UNMEASURED})

    def test_a_run_sizes_the_kv_cache_at_the_engines_count(self):
        # The mock engine reports its KV bytes per token: the fit takes them, not the model's 16-bit K and V.
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage=1, mock={"kv_bytes_per_token": 36864})
        fit = result["context"]["fit"]
        self.assertEqual(fit["kv_bytes_per_token"], 36864)
        self.assertEqual(fit["kv_bytes_at_requirement"], 36864 * result["context"]["tokens"])
        self.assertIsNone(fit["unmeasured"])
        with tempfile.TemporaryDirectory() as directory:
            fit = run(directory, stage=1)["context"]["fit"]
        self.assertEqual((fit["kv_bytes_per_token"], fit["kv_bytes_at_requirement"], fit["unmeasured"]),
                         (None, None, stages.KV_UNMEASURED))

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


class CounterTests(unittest.TestCase):
    """The engine's own expert cache counters over each speed suite (runner.Counters) and the hit rate a declared
    host_bytes_per_pass implies (runner.declared_hit_rate)."""

    class Engine:
        def __init__(self, *readings):
            self.readings = list(readings)

        def telemetry(self):
            reads, hits = self.readings.pop(0)
            return {"expert_cache": {"reads": reads, "hits": hits}}

    def test_a_window_is_the_counters_change_over_a_suite_that_completed(self):
        counters = runner.Counters(self.Engine((100, 50), (300, 200), (300, 200), (500, 300)))
        with counters.window("speed"):
            pass
        with counters.window("speed"):   # a second stretch of the same suite (a staged run's stage 2) adds to it
            pass
        self.assertEqual(counters.record(), {"speed": {"reads": 400, "hits": 250, "hit_rate": 0.625}})
        # Counters that went backwards (the engine opened again) record no window.
        backwards = runner.Counters(self.Engine((500, 300), (40, 30)))
        with backwards.window("speed"):
            pass
        self.assertEqual(backwards.record(), {})
        # A suite that raises records none: its counters cover requests it did not finish.
        failed = runner.Counters(self.Engine((0, 0), (10, 5)))
        with self.assertRaises(RuntimeError):
            with failed.window("speed"):
                raise RuntimeError("disturbed")
        self.assertEqual(failed.record(), {})
        # An engine without telemetry(), or whose telemetry names no expert cache, records none.
        none = runner.Counters(object())
        with none.window("speed"):
            pass
        self.assertEqual(none.record(), {})

    def test_a_declared_host_bytes_per_pass_implies_a_hit_rate(self):
        tiers = {"active_expert_bytes_per_token": 4 * 10**9, "resident_off_device_bytes": 10**9}
        # 2 GB per pass from host memory beyond the 1 GB of resident weights off the device: 1 of a token's 4 GB of
        # experts misses, a 75% hit rate.
        self.assertAlmostEqual(runner.declared_hit_rate({"host_bytes_per_pass": {"link": 1.5e9, "cpu": 0.5e9}}, tiers), 0.75)
        self.assertEqual(runner.declared_hit_rate({"host_bytes_per_pass": {"link": 9e9}}, tiers), 0.0)   # clamped
        self.assertIsNone(runner.declared_hit_rate({}, tiers))
        self.assertIsNone(runner.declared_hit_rate({"host_bytes_per_pass": {"link": 1e9}}, None))

    def test_a_resumed_round_answers_the_counters_from_its_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            options = dict(log=lambda message: None, context_tokens=REQUIRED, stage=1)
            config = {"facts_from": FACTS, "expert_cache": {"reads_per_token": 10, "hit_rate": 0.75}}
            first = runner.run(MOCK, "unused-model", data, data / "a.json", config, **options)
            again = runner.run(MOCK, "unused-model", data, data / "b.json", config,
                               resume=data / "a.journal.jsonl", **options)
        self.assertEqual(again["telemetry"], first["telemetry"])
        self.assertEqual(again["journal"]["live_calls"], 0)


class StatusDetailTests(unittest.TestCase):
    def test_the_quality_gate_says_honesty_was_not_checked_only_when_it_judged_nothing(self):
        sets = {"conversations/served": {"passed": True, "failures": []}}
        checked = status._quality_gate({"quality": {"status": "scored", "sets": sets, "skipped": {}, "honesty": {
            "passed": True, "tolerance": 0.001, "max_abs_logprob_diff": 0.0, "failures": []}}})
        self.assertEqual(checked["state"], "pass")
        self.assertIn("served replies scored alike within 0.0e+00", checked["detail"])
        self.assertNotIn("honesty not checked", checked["detail"])
        unchecked = status._quality_gate({"quality": {"status": "scored", "sets": sets, "skipped": {}, "honesty": {
            "passed": None, "reason": "no honesty set fits the context"}}})
        self.assertIn("honesty not checked: no honesty set fits the context", unchecked["detail"])
        # A check that failed on its top-k lists alone is no unchecked one either.
        listed = status._quality_gate({"quality": {"status": "scored", "sets": sets, "skipped": {}, "honesty": {
            "passed": False, "tolerance": 0.001, "max_abs_logprob_diff": 0.0, "failures": ["top-k lists differ"]}}})
        self.assertEqual(listed["state"], "fail")
        self.assertNotIn("honesty not checked", listed["detail"])

    def test_a_staged_run_times_its_fork_judging(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, stage=1, quality_mode="speed")
        self.assertIn("forks", result["timing"]["parts"])


class CapacityGateTests(unittest.TestCase):
    def capacity(self, fit):
        return status._context_gates({"context": {"tokens": 200000, "certified": False, "points": {}, "fit": fit}})[0]

    def test_an_uncertified_run_leaves_capacity_unmeasured_with_the_memory_it_needs(self):
        capacity = self.capacity(stages.memory_fit({"free_bytes": 2e9}, {"kv_bytes_per_token": 25000}, 200000))
        self.assertEqual(capacity["state"], "unmeasured")
        self.assertIn("2.00 GB free after the load; the KV cache at the requirement needs 5.00 GB "
                      "(25,000 bytes per token, the engine's)", capacity["detail"])

    def test_the_kv_cache_of_an_engine_that_reports_no_count_is_unmeasured(self):
        capacity = self.capacity(stages.memory_fit({"free_bytes": 2e9}, {}, 200000))
        self.assertEqual(capacity["state"], "unmeasured")
        self.assertIn("2.00 GB free after the load; the KV cache at the requirement is unmeasured: "
                      + stages.KV_UNMEASURED, capacity["detail"])
        self.assertNotIn("needs", capacity["detail"])
        unread = self.capacity(stages.memory_fit(None, {"kv_bytes_per_token": 25000}, 200000))
        self.assertNotIn("free after the load", unread["detail"])
        self.assertIn("needs 5.00 GB", unread["detail"])


if __name__ == "__main__":
    unittest.main()
