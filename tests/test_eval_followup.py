"""Eval's follow-up turns (protocol followup): measured with whatever prompt cache the engine has, and its cached path
must answer as its fresh one did."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval.limits import ceiling as model_ceiling, headroom  # noqa: E402
from crit_eval.run import chat, estimate, history, runner, stages, timing  # noqa: E402
from crit_eval.reports import compare, status  # noqa: E402
from test_eval import MEASURED, speed_part  # noqa: E402

# Each turn with its output tokens, a part of its one request (protocol requests.cases).
TURNS = [{"turn": 1, "prompt_ids": list(range(1, 101)), "output_tokens": 64},
         {"turn": 2, "prompt_ids": list(range(1, 121)), "output_tokens": 64},
         {"turn": 3, "prompt_ids": list(range(1, 141)), "output_tokens": 64}]

# A model and a device whose floors are easy to count: 10 GB of weights read at 1,000 GB/s are 10 ms, a token's
# matmuls 2 GFLOP at 100 TFLOPS are 0.02 ms, a query-key pair 1 kFLOP, and a token's KV 100 kB.
CEILING = {"matmul_flops_per_token": {"bf16": 2e9}, "attention_flops_per_pair": 1e3, "weight_bytes": 10e9}
RATES, BANDWIDTH, KV = {"bf16": 100.0}, 1000.0, 1e5


def measured_turns(cached=(0, 0, 0), new=(100, 20, 20)):
    """The follow-up part of a result as run_followup records it: prompts of 100, 120 and 140 tokens, the first
    tokens after 100, 110 and 120 ms; new None: a result measured before the run counted the new tokens."""
    rows = [{"turn": t, "prompt_tokens": p, "cached_tokens": c, "ttft_ms": ms, "prefill_tps": None, "tokens": [1]}
            | ({} if new is None else {"new_tokens": new[t - 1]})
            for t, p, c, ms in zip((1, 2, 3), (100, 120, 140), cached, (100.0, 110.0, 120.0))]
    return {"status": "measured", "turns": rows, "ttft_ms": (110.0 * 120.0) ** 0.5, "prefill_tps": None}


def priced(followup, mode="speed"):
    """A result holding the follow-up turns with what prices them: the model's arithmetic, the engine's KV bytes per
    token and a headroom with the device's rates and bandwidth."""
    return {**MEASURED, "requirements": {"exactness": "reported", "quality_mode": mode},
            "engine": {"speculative": False, "kv_bytes_per_token": KV}, "ceiling": CEILING, "followup": followup,
            "speed": speed_part({"a": {"decode_tps": 100.0, "prompt_tokens": 100}}, decode_score=100.0, complete=True),
            "sampled_speed": speed_part({"a": {"decode_tps": 90.0, "prompt_tokens": 100}}, status="measured",
                                        decode_score=90.0, complete=True),
            "headroom": {"rates": {"tflops": RATES, "sources": {}}, "bandwidth_gbs": BANDWIDTH, "gaps": [], "missing": []}}


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

    def measure(self, engine, max_context=1000, declared=None, **info):
        prepared, info = {"followup": TURNS}, {"max_context": max_context, **info}
        first = history.run_followup(engine, info, prepared, self.protocol, lambda m: None)
        if first["status"] == "measured":
            first["again"] = history.compare_followup(
                first, history.run_followup(engine, info, prepared, self.protocol, lambda m: None))
        gate = next(g for g in status._gates({"requirements": {"exactness": "reported", "quality_mode": "preserve"}, "context": {"tokens": None,
                    "reason": "-", "missing": []}, "followup": first, "engine": declared or {}}) if g["name"] == "prompt_cache")
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

    def test_a_declared_cache_that_serves_no_turn_in_either_pass_leaves_the_gate_unmeasured(self):
        # The cached path never ran: a pass would say it answers as the fresh one without having seen it.
        first, gate = self.measure(Engine(), declared={"prompt_cache": True})
        self.assertEqual(gate["state"], "unmeasured")
        self.assertIn("no follow-up turn was served from it in either pass", gate["detail"])
        # One cached turn, in either pass, measures it; an engine that declares nothing is judged as before.
        self.assertEqual(self.measure(Engine(exact=True), declared={"prompt_cache": True})[1]["state"], "pass")
        self.assertEqual(self.measure(Engine())[1]["state"], "pass")
        # A cache that serves nothing yet answers otherwise the second time still fails.
        drifting = Engine(drifts=True)
        drifting.generate = lambda prompt, n, on_tokens, prompt_cache=False, seen=[]: (
            seen.append(1), on_tokens([len(seen)] * n),
            {"tokens": [len(seen)] * n, "served": {"prompt_tokens": len(prompt), "output_tokens": n,
                                                   "cached_tokens": 0, "finish_reason": "length"}})[-1]
        self.assertEqual(self.measure(drifting, declared={"prompt_cache": True})[1]["state"], "fail")

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

    def test_each_turn_records_the_tokens_past_the_prefix_it_shares_with_the_turn_before(self):
        first, _ = self.measure(Engine())
        self.assertEqual([t["new_tokens"] for t in first["turns"]], [100, 20, 20])
        # A turn whose prompt parts from the earlier one midway (a template that ends a prompt differently): only the
        # shared prefix is what a cache holds.
        self.assertEqual(history.new_tokens([1, 2, 3, 9, 9], [1, 2, 3, 4]), 2)
        self.assertEqual(history.new_tokens([1, 2, 3], None), 3)
        self.assertEqual(history.new_tokens([1, 2], [1, 2, 3]), 0)


class FollowupFloorTests(unittest.TestCase):
    """The follow-up turns' floor with a prompt cache (headroom.followup_summary) and the goal status states."""
    protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")

    def test_a_cached_prefix_leaves_the_new_tokens_arithmetic_and_the_whole_prompts_kv(self):
        fresh, _ = model_ceiling.ttft_floor_ms(CEILING, RATES, BANDWIDTH, 120, kv_bytes_per_token=KV)
        same, _ = model_ceiling.ttft_floor_ms(CEILING, RATES, BANDWIDTH, 120, kv_bytes_per_token=KV, cached_tokens=0)
        self.assertEqual(fresh, same)
        cached, _ = model_ceiling.ttft_floor_ms(CEILING, RATES, BANDWIDTH, 120, kv_bytes_per_token=KV, cached_tokens=100)
        # 20 tokens' matmuls, each attending the 100 cached tokens and the new ones up to itself.
        pairs = 20 * 100 + 20 * 21 / 2
        self.assertAlmostEqual(cached["compute_ms"], (20 * 2e9 + pairs * 1e3) / 100e12 * 1000)
        self.assertLess(cached["compute_ms"], fresh["compute_ms"])
        # The KV bytes are the whole prompt's: the cached tokens' are read where the new ones' are written.
        self.assertAlmostEqual(cached["memory_ms"], (10e9 + 120 * KV) / (BANDWIDTH * 1e9) * 1000)
        self.assertEqual(cached["memory_ms"], fresh["memory_ms"])
        self.assertEqual((cached["bound"], cached["total_ms"]), ("memory", cached["memory_ms"]))

    def test_the_floor_is_each_later_turn_with_the_turn_before_it_cached(self):
        found = headroom.followup_summary(priced(measured_turns()))
        self.assertEqual(sorted(found["turns"]), ["2", "3"])   # the first turn has nothing to reuse
        floors = [(10e9 + n * KV) / (BANDWIDTH * 1e9) * 1000 for n in (120, 140)]
        self.assertAlmostEqual(found["turns"]["2"]["floor"]["total_ms"], floors[0])
        self.assertAlmostEqual(found["floor_ms"], (floors[0] * floors[1]) ** 0.5)
        self.assertAlmostEqual(found["score_ms"], (110.0 * 120.0) ** 0.5)
        self.assertAlmostEqual(found["fraction_of_floor"], found["floor_ms"] / found["score_ms"])
        gap = found["gap"]
        self.assertEqual((gap["kpi"], gap["name"]), ("followup_ttft", headroom.FOLLOWUP_GAP))
        self.assertAlmostEqual(gap["projected_gain"], found["score_ms"] / found["floor_ms"] - 1)
        self.assertIn("turn 2: 110 vs 10.0 ms (20 of 120 prompt tokens new, 0 served from a cache; memory bound)",
                      gap["detail"])

    def test_a_result_measured_before_the_run_counted_new_tokens_takes_the_prompts_growth(self):
        found = headroom.followup_summary(priced(measured_turns(new=None)))
        self.assertEqual([found["turns"][t]["new_tokens"] for t in ("2", "3")], [20, 20])

    def test_a_cache_that_held_more_leaves_less_and_a_turn_served_whole_has_no_floor(self):
        # The engine reports 110 of turn 2's 120 tokens cached, more than the 100 the turns share: 10 are computed.
        more = headroom.followup_summary(priced(measured_turns(cached=(0, 110, 120))))
        self.assertEqual(more["turns"]["2"]["new_tokens"], 10)
        whole = headroom.followup_summary(priced(measured_turns(cached=(0, 120, 120))))
        self.assertNotIn("floor", whole["turns"]["2"])
        self.assertEqual((whole["floor_ms"], whole["gap"]), (None, None))   # no floor for the KPI without every turn's

    def test_turns_that_were_not_measured_have_no_floor(self):
        self.assertIsNone(headroom.followup_summary(priced({"status": "skipped", "reason": "-"})))
        self.assertIsNone(headroom.followup_summary(priced(None)))
        # Without the device's rates the floor is unknown and says what is missing.
        bare = priced(measured_turns())
        bare["headroom"]["rates"] = None
        found = headroom.followup_summary(bare)
        self.assertEqual((found["floor_ms"], found["missing"]), (None, ["device GEMM rates (--device)"]))

    def test_status_states_the_goal_beside_the_runs_own_and_leaves_the_stop_record_alone(self):
        current = status.build(priced(measured_turns()), self.protocol)
        goal = current["goals"][-1]
        floor = headroom.followup_summary(priced(measured_turns()))["floor_ms"]
        self.assertEqual((goal["kpi"], goal["unit"]), ("followup_ttft", "ms"))
        self.assertAlmostEqual(goal["limit"], floor)
        self.assertAlmostEqual(goal["fraction_of_limit"], floor / goal["score"])
        kpi = current["kpis"]["followup"]
        self.assertAlmostEqual(kpi["floor_ms"], floor)
        self.assertEqual((kpi["turns"]["2"]["new_tokens"], kpi["turns"]["2"]["bound"]), (20, "memory"))
        self.assertNotIn("new_tokens", kpi["turns"]["1"])
        # A result measured before the run priced the turns gets the gap now, once.
        self.assertEqual([g["name"] for g in current["gaps"] if g["kpi"] == "followup_ttft"], [headroom.FOLLOWUP_GAP])
        # crit run's stop record and priority rule take the KPIs they know: the goal is stated, not ranked.
        self.assertNotIn("followup_ttft", [row["kpi"] for row in current["stop_record_headroom"]["kpis"]])
        text = status.render(current)
        self.assertIn("followup_ttft 115 -> 10 ms (-91%; 8.7% of its limit)", text)
        self.assertIn("followup 115 ms to the first token of a follow-up turn (geometric mean over 2 turns), 8.7% of its "
                      "10.0 ms floor with a prompt cache; turn 2 110 ms vs 10.0 ms (memory bound), 0 of 120 prompt tokens "
                      "served from a cache, 20 new", text)
        self.assertIn("followup_ttft +1047% followup_ttft score", text)

    def test_the_run_prices_the_turns_into_the_headroom_and_status_reads_them_there(self):
        result = priced(measured_turns())
        result["headroom"] = headroom.price_prompts(result, RATES, {}, [])
        room = result["headroom"]
        self.assertAlmostEqual(room["followup"]["floor_ms"], headroom.followup_summary(result)["floor_ms"])
        self.assertNotIn("gap", room["followup"])
        self.assertEqual([g["kpi"] for g in room["gaps"]], ["followup_ttft"])
        # Priced again (crit eval status --device), the gap is replaced, never added twice.
        result["headroom"] = headroom.price_prompts(result, RATES, {}, [])
        current = status.build(result, self.protocol)
        self.assertEqual([g["kpi"] for g in current["gaps"]], ["followup_ttft"])
        self.assertEqual(current["goals"][-1]["kpi"], "followup_ttft")

    def test_an_engine_without_measured_turns_gets_no_goal(self):
        current = status.build(priced({"status": "skipped", "no_prompt_cache": True, "reason": "-"}), self.protocol)
        self.assertNotIn("followup_ttft", [goal["kpi"] for goal in current["goals"]])
        self.assertIsNone(current["kpis"]["followup"])
        self.assertNotIn("followup", status.render(current))

    def test_an_engine_with_a_prompt_cache_stands_near_its_floor(self):
        # A cache serving each turn's shared prefix, the first tokens after 12 and 13 ms: most of the limit reached.
        turns = measured_turns(cached=(0, 100, 120))
        for row, ms in zip(turns["turns"][1:], (12.0, 13.0)):
            row["ttft_ms"] = ms
        turns["ttft_ms"] = (12.0 * 13.0) ** 0.5
        goal = status.build(priced(turns), self.protocol)["goals"][-1]
        self.assertGreater(goal["fraction_of_limit"], 0.75)
        self.assertLess(goal["fraction_of_limit"], 1.0)

    def test_compare_lists_the_turns_against_their_floor(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = {**priced(measured_turns()), "model_sha256": "m" * 64, "suite_sha256": "s" * 64,
                      "protocol": self.protocol["protocol"], "host": {"platform": "test"},
                      "engine": {"name": "toy", "speculative": False, "kv_bytes_per_token": KV}}
            path.write_text(json.dumps(result), encoding="utf-8")
            text = compare.compare([path, path])
        self.assertIn("Follow-up TTFT (geomean ms)", text)
        self.assertIn("115 (8.7% of 10.0 ms floor with a prompt cache)", text)


if __name__ == "__main__":
    unittest.main()
