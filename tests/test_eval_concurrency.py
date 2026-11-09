"""The concurrency part (protocol concurrency): closed-loop clients against a fake server that runs several requests at
once, the levels it measures, its aggregate, peak, per-request and consistency figures, the concurrency gate, its
budget, journal and status, and the batched ceiling."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval.run import adapters, chat, concurrency, runner  # noqa: E402
from crit_eval.limits import batched, tiers  # noqa: E402
from crit_eval.quality import prepare  # noqa: E402
from crit_eval.reports import compare, status  # noqa: E402
from test_eval_server import REPLY, Text, write_data  # noqa: E402

PROTOCOL = json.loads((ROOT / "crit_eval/protocol.json").read_text(encoding="utf-8"))


class BatchingChat(BaseHTTPRequestHandler):
    """A server that runs requests at once: each streams REPLY's first max_tokens characters, one per `step` seconds,
    whatever else runs. With `unstable`, a reply's characters from the sixth on come out upper case while another
    request runs beside it (an engine whose batched arithmetic is not its serial arithmetic)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == "/health" and self.server.telemetry:   # crit serve's engine counters (engine_telemetry)
            cache = {"reads": self.server.telemetry["expert_reads"], "hits": self.server.telemetry["expert_hits"]}
            self.wfile.write(json.dumps({"engine_telemetry": {"expert_cache": cache}}).encode())
            return
        self.wfile.write(json.dumps({"object": "list", "data": [{"id": "fake", **self.server.card}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.server.serial:   # one request at a time, whatever the card says
            self.server.one.acquire()
        with self.server.lock:
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
        try:
            reply = REPLY[:body["max_tokens"]]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index, piece in enumerate([""] + list(reply)):
                if self.server.unstable and index > 5 and self.server.active > 1:
                    piece = piece.upper()
                event = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(self.server.step)
            usage = {"prompt_tokens": 20, "completion_tokens": len(reply), **self.server.usage}
            for event in ({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                          {"choices": [], "usage": usage, **({"crit_telemetry": self.server.telemetry}
                                                             if self.server.telemetry else {})}):
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            with self.server.lock:
                self.server.active -= 1
            if self.server.serial:
                self.server.one.release()


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), BatchingChat)
        self.server.card = {"max_concurrent_requests": 4}
        self.server.lock, self.server.active, self.server.peak = threading.Lock(), 0, 0
        self.server.step, self.server.unstable, self.server.usage = 0.002, False, {}
        self.server.serial, self.server.one, self.server.telemetry = False, threading.Lock(), None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        patcher = patch.object(chat, "load_text", lambda model, chat_template=None: Text())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.data = Path(tempfile.mkdtemp())
        write_data(self.data)
        self.protocol = json.loads(json.dumps(PROTOCOL))
        self.protocol["concurrency"]["output_tokens"] = 24   # CI-sized levels
        self.protocol["concurrency"]["requests_per_client"] = 2

    def engine(self, context_tokens=4096):
        prepared = json.loads((self.data / "prepared.json").read_text())
        return adapters.open_engine(chat.Server(self.url, prepared), "unused-model", {}, True, context_tokens)

    def measure(self, result=None, fits=None, inputs=None, required=4096):
        prepared = json.loads((self.data / "prepared.json").read_text())
        engine, info = self.engine()
        serial = result if result is not None else {"speed": {"cases": {
            "chat": {"status": "measured", "tokens": prepare.token_ids(Text(), REPLY[:12])}}}}
        return concurrency.run_concurrency(engine, info, prepared, self.protocol, serial, required, lambda m: None,
                                           fits, inputs)

    def test_levels_measure_the_aggregate_peak_first_tokens_and_consistency(self):
        self.server.step = 0.004
        found = self.measure()
        self.assertEqual((found["status"], found["planned"], found["max_concurrent_requests"]), ("measured", [1, 2, 4], 4))
        self.assertEqual(found["pool"], ["chat", "tool-call", "heldout/de"])
        for clients in (1, 2, 4):
            level = found["levels"][str(clients)]
            self.assertTrue(level["qualified"], level["errors"])
            self.assertEqual((level["peak_concurrency"], level["requests"], level["completion_tokens"]),
                             (clients, 2 * clients, 2 * clients * 24))
            # The replies streamed at once: the server ran the requests together (one client runs beside nothing).
            self.assertEqual(level["ran_together"], True if clients > 1 else None)
            self.assertGreaterEqual(level["streaming_peak"], min(2, clients))
            self.assertAlmostEqual(level["speedup"], level["aggregate_tps"] / found["levels"]["1"]["aggregate_tps"])
            self.assertGreater(level["ttft_ms_p95"], 0)
            self.assertGreater(level["decode_tps_p50"], 0)
            # Every chat reply equals the speed part's serial reply as far as that went.
            self.assertEqual(level["consistency"]["fraction"], 1.0)
            self.assertGreater(level["consistency"]["compared"], 0)
        # The server runs the clients at once: four of them deliver well over twice one's tokens per second.
        self.assertGreater(found["levels"]["4"]["aggregate_tps"], 2 * found["levels"]["1"]["aggregate_tps"])
        self.assertEqual(self.server.peak, 4)

    def test_replies_that_differ_under_batching_are_reported_with_their_first_difference(self):
        self.server.unstable = True
        level = self.measure()["levels"]["2"]
        self.assertTrue(level["qualified"])
        self.assertLess(level["consistency"]["fraction"], 1.0)
        first = {d["first_difference"] for d in level["consistency"]["differences"].values()}
        self.assertTrue(first and min(first) >= 5, first)

    def gate(self, found, mode="preserve"):
        return status._concurrency_gate({"concurrency": found, "requirements": {"quality_mode": mode}})

    def test_the_gate_passes_an_engine_whose_batched_replies_are_its_serial_ones_and_scale(self):
        self.server.step = 0.004
        gate = self.gate(self.measure())
        self.assertEqual(gate["state"], "pass", gate["detail"])
        self.assertIn("batched replies equal their serial ones", gate["detail"])

    def test_the_gate_fails_replies_that_change_under_batching(self):
        self.server.unstable = True
        gate = self.gate(self.measure())
        self.assertEqual(gate["state"], "fail")
        self.assertIn("differs from its serial reply at token", gate["detail"])

    def test_the_gate_fails_a_server_that_runs_its_clients_one_after_another(self):
        # Its clients see their requests in flight together, queued at the server: only the replies show it streamed
        # one at a time.
        self.server.serial, self.server.step = True, 0.004
        found = self.measure()
        self.assertEqual(self.server.peak, 1)
        for clients in ("2", "4"):
            level = found["levels"][clients]
            self.assertEqual((level["qualified"], level["peak_concurrency"]), (True, int(clients)))
            self.assertEqual((level["ran_together"], level["streaming_peak"]), (False, 1))
        gate = self.gate(found)
        self.assertEqual(gate["state"], "fail")
        self.assertIn("2 clients did not run together: at most 1 of their 2 replies streamed at a time", gate["detail"])

    def test_replies_streaming_at_once_are_counted_by_their_overlap(self):
        row = lambda first, last: {"first": first, "last": last}   # noqa: E731
        self.assertEqual(concurrency.streaming_peak([row(0, 1), row(1, 2), row(2, 3)]), 1)   # one after another
        self.assertEqual(concurrency.streaming_peak([row(0, 2), row(1, 3), row(1.5, 4)]), 3)
        # A reply of one chunk streams over no interval; a request without a reply is no stream.
        self.assertEqual(concurrency.streaming_peak([row(0, 0), row(0, 0), {"first": None, "last": None}]), 0)
        # Two or more at once ran together, one at a time did not, none at all shows nothing (each in one chunk).
        self.assertEqual([concurrency.together(4, peak) for peak in (0, 1, 2, 4)], [None, False, True, True])
        self.assertIsNone(concurrency.together(1, 1))

    def test_replies_in_one_chunk_leave_the_gate_unmeasured(self):
        # A server that sends each reply in one chunk: no reply streams over an interval, so nothing shows whether the
        # requests ran together.
        rows = [{"first": 0.5, "last": 0.5}, {"first": 0.6, "last": 0.6}]
        self.assertEqual(concurrency.together(2, concurrency.streaming_peak(rows)), None)
        level = lambda peak, together: {"qualified": True, "aggregate_tps": 9.0, "errors": [], "ran_together": together,  # noqa: E731
                                        "streaming_peak": peak, "consistency": {"compared": 2, "matched": 2,
                                                                                "differences": {}}}
        gate = self.gate({"status": "measured", "skipped": {}, "levels": {"1": level(0, None), "2": level(0, None)}})
        self.assertEqual(gate["state"], "unmeasured")
        self.assertIn("2 clients: no reply streamed over an interval (each came in one chunk)", gate["detail"])
        # Another condition that fails still fails it.
        gate = self.gate({"status": "measured", "skipped": {}, "levels": {
            "1": level(0, None), "2": level(0, None), "4": {**level(1, False), "qualified": False,
                                                            "errors": ["Expected requests are missing"]}}})
        self.assertEqual(gate["state"], "fail")

    def test_without_the_speed_part_each_case_runs_alone_first_and_is_compared(self):
        # crit eval run --only concurrency: no speed part's replies, so the part runs each pool speed case alone.
        self.server.step = 0.004
        found = self.measure(result={})
        self.assertEqual(found["serial_source"], "one-client runs")
        for level in found["levels"].values():
            self.assertGreater(level["consistency"]["compared"], 0)
            self.assertEqual(level["consistency"]["fraction"], 1.0)
        self.assertEqual(self.gate(found)["state"], "pass")
        self.server.unstable = True
        self.assertEqual(self.gate(self.measure(result={}))["state"], "fail")

    def test_the_gate_does_not_pass_on_no_reply_compared(self):
        level = lambda tps, peak: {"qualified": True, "aggregate_tps": tps, "errors": [], "streaming_peak": peak,  # noqa: E731
                                   "ran_together": concurrency.together(peak, peak),
                                   "consistency": {"compared": 0, "matched": 0, "differences": {}}}
        found = {"status": "measured", "skipped": {}, "levels": {"1": level(100.0, 1), "4": level(390.0, 4)}}
        gate = self.gate(found)
        self.assertEqual(gate["state"], "unmeasured")
        self.assertIn("no batched reply compared", gate["detail"])

    def test_the_gate_judges_replies_and_qualification_never_the_speedup(self):
        def level(tps, differences=None, together=True, peak=2):
            return {"qualified": True, "aggregate_tps": tps, "errors": [], "ran_together": together,
                    "streaming_peak": peak, "consistency": {"compared": 2, "matched": 2 - len(differences or {}),
                                                            "differences": differences or {}}}
        # An engine whose expert misses cross the host link serves little more at two clients: it passes, its speedup a
        # KPI against the cost model's.
        found = {"status": "measured", "skipped": {}, "levels": {"1": level(100.0, together=None, peak=1),
                                                                 "2": level(104.0)}}
        gate = self.gate(found)
        self.assertEqual(gate["state"], "pass", gate["detail"])
        self.assertIn("replies streaming at once at the peak: 2 clients 2 of 2", gate["detail"])
        self.assertNotIn("1.04", gate["detail"])
        found["levels"]["4"] = level(30.0, together=False, peak=1)
        gate = self.gate(found)
        self.assertEqual(gate["state"], "fail")
        self.assertIn("4 clients did not run together", gate["detail"])
        found["levels"]["4"] = level(390.0, peak=4)
        self.assertEqual(self.gate(found)["state"], "pass")
        # A difference at a near-tie passes in speed only; a decisive fork fails in either mode.
        tie = {"0:1": {"case": "chat", "match": False, "first_difference": 7,
                       "fork": {"position": 7, "tokens": [3, 4], "logprobs": [-0.5, -0.9], "gap": 0.4, "near_tie": True}}}
        found["levels"]["2"] = level(200.0, tie)
        self.assertEqual(self.gate(found, "speed")["state"], "pass")
        self.assertEqual(self.gate(found)["state"], "fail")
        tie["0:1"]["fork"].update(gap=4.0, near_tie=False)
        self.assertEqual(self.gate(found, "speed")["state"], "fail")
        unqualified = {**level(9.0), "qualified": False, "aggregate_tps": None, "errors": ["Expected requests are missing"]}
        self.assertEqual(self.gate({"status": "measured", "skipped": {},
                                    "levels": {"1": level(1.0, together=None), "2": unqualified}})["state"], "fail")
        # A part that did not measure, or measured one client alone, leaves it unmeasured.
        self.assertEqual(self.gate({"status": "unmeasured", "reason": "x"})["state"], "unmeasured")
        self.assertEqual(self.gate({"status": "measured", "skipped": {},
                                    "levels": {"1": level(1.0, together=None)}})["state"], "unmeasured")

    def test_cached_prompt_tokens_disqualify_a_level(self):
        self.server.usage = {"prompt_tokens_details": {"cached_tokens": 8}}
        level = self.measure()["levels"]["1"]
        self.assertFalse(level["qualified"])
        self.assertIsNone(level["aggregate_tps"])
        self.assertTrue(any("completion failed validation" in e for e in level["errors"]), level["errors"])

    def test_a_server_that_runs_one_request_or_reports_none_is_unmeasured(self):
        self.server.card = {"max_concurrent_requests": 1}
        found = self.measure()
        self.assertEqual((found["status"], found["max_concurrent_requests"]), ("unmeasured", 1))
        self.server.card = {}
        self.assertIn("reports no max_concurrent_requests", self.measure()["reason"])

    def test_levels_stop_at_the_capacity_and_the_kv_their_requests_hold(self):
        prompts = [{"id": "a", "prompt_ids": [1] * 100, "case": "a"}, {"id": "b", "prompt_ids": [1] * 120, "case": None}]
        rule = self.protocol["concurrency"]
        # The longest request holds 120 + 24 tokens: 32 of them fit 8,192 tokens of KV, 16 fit 4,096.
        self.assertEqual(concurrency.levels(self.protocol, 64, 8192, prompts), [1, 2, 4, 8, 16, 32])
        self.assertEqual(concurrency.levels(self.protocol, 64, 4096, prompts), [1, 2, 4, 8, 16])
        self.assertEqual(concurrency.levels(self.protocol, 3, 1 << 20, prompts), [1, 2])
        self.assertEqual(rule["clients"], [1, 2, 4, 8, 16, 32, 64])
        # The pool: speed cases up to max_prompt_tokens, then the held-out prompts, each fitting the context.
        prepared = json.loads((self.data / "prepared.json").read_text())
        self.assertEqual([p["id"] for p in concurrency.pool(prepared, self.protocol, 4096)], ["chat", "tool-call", "heldout/de"])
        self.assertEqual(concurrency.pool(prepared, self.protocol, 30), [])

    def test_a_budget_stops_the_sweep_at_the_first_level_that_would_pass_it(self):
        asked = []

        def fits(name, seconds):
            asked.append((name, seconds))
            return name != "concurrency@4"
        found = self.measure(fits=fits)
        self.assertEqual(list(found["levels"]), ["1", "2"])
        self.assertIn("4", found["skipped"])
        # The first level has no rate to price it; the next ones are priced from the best aggregate so far.
        self.assertIsNone(asked[0][1])
        self.assertTrue(all(seconds > 0 for _, seconds in asked[1:]))

    def test_each_level_is_priced_against_the_batched_ceiling(self):
        terms = {"resident_bytes": 1e9, "expert_bytes": 0, "experts": 0, "experts_per_token": 0,
                 "dense_flops": {"bf16": 2e9}, "expert_flops": {}, "kv_bytes_per_token": 1000, "fixed_state_bytes": 0,
                 "attention_flops_per_pair": 0}
        found = self.measure(inputs=lambda: (terms, 1000.0, {"bf16": 100.0}, None))
        for clients, level in found["levels"].items():
            floor = level["ceiling"]
            self.assertEqual(floor["rows"], int(clients))
            self.assertAlmostEqual(level["fraction_of_ceiling"], level["aggregate_tps"] / floor["ceiling_tps"])
        unpriced = self.measure(inputs=lambda: (None, None, None, None))["levels"]["1"]
        self.assertIsNone(unpriced["ceiling"])
        self.assertIsNone(unpriced["expected_speedup"])
        self.assertIn("no batched ceiling", unpriced["expected_missing"])

    def test_each_level_reports_its_speedup_beside_the_cost_models(self):
        # A dense model read once per step for every row: the cost model expects close to N times one client.
        terms = {"resident_bytes": 1e9, "expert_bytes": 0, "experts": 0, "experts_per_token": 0,
                 "dense_flops": {"bf16": 2e9}, "expert_flops": {}, "kv_bytes_per_token": 1000, "fixed_state_bytes": 0,
                 "attention_flops_per_pair": 0}
        self.server.step = 0.004
        found = self.measure(inputs=lambda: (terms, 1000.0, {"bf16": 100.0}, None))
        one = found["levels"]["1"]
        self.assertEqual((one["speedup"], one["expected_speedup"]), (1.0, 1.0))
        for clients in ("2", "4"):
            level = found["levels"][clients]
            expected = level["ceiling"]["ceiling_tps"] / one["ceiling"]["ceiling_tps"]
            self.assertAlmostEqual(level["expected_speedup"], expected)
            self.assertGreater(level["expected_speedup"], 0.9 * int(clients))
            self.assertAlmostEqual(level["fraction_of_expected"], level["speedup"] / expected)
        # The status reports the speedup as a KPI beside its expectation; the gate does not read it.
        result = {"concurrency": found, "requirements": {"quality_mode": "preserve"}, "engine": {}}
        kpi = status._concurrency_kpi(result)
        self.assertAlmostEqual(kpi["levels"]["2"]["expected_speedup"], found["levels"]["2"]["expected_speedup"])
        text = "\n".join(status._render_concurrency(kpi))
        self.assertIn("speedup over one client's aggregate (reported, not judged): 2 clients", text)
        self.assertIn(f"of {found['levels']['2']['expected_speedup']:.2f}x expected (", text)
        self.assertNotIn("no expectation", text)
        # compare shows the speedup beside each level's aggregate.
        from crit_eval.run import host
        (self.data / "result.json").write_text(json.dumps({"label": "x", "engine": {"name": "fake"}, "host": host.describe(),
                                                           "model_sha256": "m", "suite_sha256": "s", "concurrency": found,
                                                           "requirements": {"quality_mode": "preserve"}}))
        table = compare.compare([self.data / "result.json"])
        self.assertIn(f"{found['levels']['2']['speedup']:.2f}x one client ({found['levels']['2']['fraction_of_expected']:.0%} "
                      f"of {found['levels']['2']['expected_speedup']:.2f}x expected)", table)

    @staticmethod
    def offloaded(rates=None):
        """A model with most experts in host memory (64 GB of experts, 24 GB of them on the device), its tiers as
        crit_eval.limits.tiers.derive gives them: link 28 GB/s, host memory 45 GB/s, a CPU of 1,000 GFLOPS."""
        gb = 1e9
        terms = {"resident_bytes": 4 * gb, "expert_bytes": 64 * gb, "experts": 512, "experts_per_token": 10,
                 "dense_flops": {"bf16": 2e9}, "expert_flops": {"nvfp4": 2 * 64e9}, "kv_bytes_per_token": 1000,
                 "fixed_state_bytes": 0, "attention_flops_per_pair": 0}
        where = {"scenario": "sparse_device_host", "tokens": 8192,
                 "bytes": {"resident": 4 * 10**9, "resident_off_device": 0, "experts": 64 * 10**9,
                           "experts_on_device": 24 * 10**9, "experts_off_device": 40 * 10**9},
                 "active_expert_bytes_per_token": 64 * 10**9 * 10 // 512,
                 "rates_gbs": rates or {"device": 1800, "link": 28, "host_memory_read": 45}, "cpu_gflops": 1000}
        return terms, tiers.derive(where, {"matmul_flops_per_token": {"nvfp4": 2.5e9}, "weight_bytes": 2.5e9})

    def test_misses_from_host_memory_bound_the_speedup_each_row_brings_its_own(self):
        # An engine with most experts in host memory: at a measured 64% hit rate each row's misses come from host
        # memory, so the cost model expects about one client's aggregate at two, and an engine serving 1.04x reaches
        # most of it. The host bytes take the faster of the link and the CPU.
        gb = 1e9
        terms, tiered = self.offloaded()
        host = {"tiers": tiered, "hit_rate": 0.64, "hit_rate_source": "engine"}
        rates = {"bf16": 200.0, "nvfp4": 800.0}
        one, _ = batched.step_floor(terms, 1, 1000, 1800.0, rates, host)
        two, _ = batched.step_floor(terms, 2, 1000, 1800.0, rates, host)
        self.assertEqual((one["bound"], two["bound"], one["hit_rate_source"]), ("cpu", "cpu", "engine"))
        self.assertAlmostEqual(one["host_bytes"], 64 * gb * 10 / 512 * 0.36, delta=1e3)
        self.assertAlmostEqual(one["host_paths_ms"]["link"], one["host_bytes"] / 28e9 * 1000)
        self.assertAlmostEqual(one["host_ms"], one["host_bytes"] / 45e9 * 1000)   # host memory reads faster than the link
        self.assertLess(two["ceiling_tps"] / one["ceiling_tps"], 1.05)
        # A link faster than host memory carries them; a slow CPU's arithmetic on many rows makes the link win too.
        _, fast = self.offloaded({"device": 1800, "link": 60, "host_memory_read": 45})
        self.assertEqual(batched.step_floor(terms, 1, 1000, 1800.0, rates, {**host, "tiers": fast})[0]["host_path"],
                         "link")
        slow = {**tiered, "cpu_gflops": 1.0}
        self.assertEqual(batched.step_floor(terms, 8, 1000, 1800.0, rates, {**host, "tiers": slow})[0]["host_path"],
                         "link")
        # Resident weights off the device beyond those a step reads (the placement counts the draft layers too) never
        # make the device's bytes negative.
        over = batched.step_floor(terms, 1, 1000, 1800.0, rates,
                                  {**host, "tiers": {**tiered, "resident_off_device_bytes": 9 * 10**9}})[0]
        self.assertGreaterEqual(over["memory_ms"], 0)
        self.assertEqual(over["host_bytes"], int(4 * gb + one["host_bytes"]))
        # All on the device the same model expects nearly twice as much.
        plain_one, _ = batched.step_floor(terms, 1, 1000, 1800.0, rates)
        plain_two, _ = batched.step_floor(terms, 2, 1000, 1800.0, rates)
        self.assertGreater(plain_two["ceiling_tps"] / plain_one["ceiling_tps"], 1.5)
        # 64 rows touch more experts than the device holds: at least the excess crosses, whatever the hit rate says.
        many, _ = batched.step_floor(terms, 64, 1000, 1800.0, rates, {**host, "hit_rate": 0.99})
        self.assertEqual(many["hit_rate_source"], "device capacity")
        self.assertAlmostEqual(many["host_bytes"], 64 * gb * batched.touched(512, 10, 64) - 24 * gb, delta=1e3)
        # Without a measured hit rate the misses beyond that bound are unmeasured: the ceiling holds the bound alone
        # and the level has no expectation.
        bare, _ = batched.step_floor(terms, 2, 1000, 1800.0, rates, {**host, "hit_rate": None})
        self.assertEqual((bare["hit_rate"], bare["hit_rate_source"], bare["host_bytes"]), (None, "unmeasured", 0))
        self.assertIn("expert_reads and expert_hits", bare["unmeasured"])
        at = lambda floor: {k: floor.get(k) for k in ("ceiling_tps", "bound", "hit_rate", "hit_rate_source",  # noqa: E731
                                                        "unmeasured")}
        levels = {"1": {"aggregate_tps": 30.0, "expected_at": {**at(bare), "ceiling_tps": 100.0}, "expected_at_missing": None},
                  "2": {"aggregate_tps": 31.0, "expected_at": {**at(bare), "ceiling_tps": 190.0}, "expected_at_missing": None}}
        concurrency.expectations(levels)
        self.assertIsNone(levels["2"]["expected_speedup"])
        self.assertIn("expert_reads and expert_hits", levels["2"]["expected_missing"])
        self.assertAlmostEqual(levels["2"]["speedup"], 31 / 30)
        # Host bytes without the probe's link and host memory rates are a missing input, not a guess.
        unprobed = {**tiered, "rates_gbs": {"link": None, "memory": None}}
        self.assertEqual(batched.step_floor(terms, 1, 1000, 1800.0, rates, {**host, "tiers": unprobed}),
                         (None, ["host link and host memory rates (re-run crit eval probe)"]))

    def test_a_level_whose_requests_report_expert_hits_is_priced_at_its_own_hit_rate(self):
        # Its ceiling (and so its fraction of it) at its own requests' hit rate; its expectation at the run's
        # single-stream rate, as every level's, so the expected speedups compare like with like.
        terms, tiered = self.offloaded()
        rates = {"bf16": 200.0, "nvfp4": 800.0}
        self.server.telemetry = {"expert_reads": 100, "expert_hits": 80}
        run = {"tiers": tiered, "hit_rate": 0.64, "hit_rate_source": "engine"}
        found = self.measure(inputs=lambda: (terms, 1800.0, rates, run))
        level = found["levels"]["2"]
        self.assertEqual(level["expert_hit_rate"], 0.8)
        self.assertEqual((level["ceiling"]["hit_rate"], level["ceiling"]["hit_rate_source"]), (0.8, "level requests"))
        self.assertEqual((level["expected_at"]["hit_rate"], level["expected_at"]["hit_rate_source"]), (0.64, "engine"))
        at = lambda rows: batched.step_floor(terms, rows, found["mean_context"], 1800.0, rates, run)[0]["ceiling_tps"]  # noqa: E731
        self.assertAlmostEqual(level["expected_speedup"], at(2) / at(1))
        self.assertAlmostEqual(level["fraction_of_ceiling"], level["aggregate_tps"] / level["ceiling"]["ceiling_tps"])
        self.assertEqual((found["hit_rate"], found["hit_rate_source"]), (0.64, "engine"))
        # The status names every level's ceiling's hit rate and the run's the expectations are priced at.
        text = "\n".join(status._render_concurrency(status._concurrency_kpi({"concurrency": found, "engine": {}})))
        self.assertIn("1 clients 80% (level requests), 2 clients 80% (level requests)", text)
        self.assertIn("at the run's single-stream 64% expert cache hit rate (engine)", text)
        # Without a run's rate the expectation is unmeasured even where a level's requests report hits.
        bare = self.measure(inputs=lambda: (terms, 1800.0, rates, {**run, "hit_rate": None,
                                                                    "hit_rate_source": "unmeasured"}))
        self.assertIsNone(bare["levels"]["2"]["expected_speedup"])
        self.assertIn("expert_reads and expert_hits", bare["levels"]["2"]["expected_missing"])
        text = "\n".join(status._render_concurrency(status._concurrency_kpi({"concurrency": bare, "engine": {}})))
        self.assertNotIn("optimistic", text)   # the levels' own ceilings are priced at their requests' hit rates
        # A served engine's own counters, from crit serve's /health.
        engine, _ = self.engine()
        self.assertEqual(engine.telemetry(), {"expert_cache": {"reads": 100, "hits": 80}})
        self.server.telemetry = None
        self.assertIsNone(engine.telemetry())

    def test_a_run_measures_concurrency_after_the_speed_part_and_status_reports_it(self):
        self.server.step = 0.0005
        protocol = self.protocol
        with patch.object(runner, "load_json", lambda path: protocol if Path(path).name == "protocol.json"
                          else json.loads(Path(path).read_text(encoding="utf-8"))):
            result = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "concurrency"),
                                warmups=0, repetitions=1, log=lambda m: None, context_tokens=4096, url=self.url + "/",
                                stage="full")
        self.assertEqual(result["concurrency"]["status"], "measured")
        self.assertEqual(list(result["concurrency"]["levels"]), ["1", "2", "4"])
        self.assertIn("concurrency", result["timing"]["parts"])
        gates = {g["name"]: g for g in result["status"]["gates"]}
        self.assertEqual(gates["concurrency"]["state"], "pass", gates["concurrency"]["detail"])
        kpi = result["status"]["kpis"]["concurrency"]
        self.assertEqual(kpi["levels"]["4"]["consistency"], 1.0)
        text = status.render(result["status"])
        self.assertIn("concurrency 3 levels up to 4 clients", text)
        self.assertIn("4 clients:", text)
        self.assertIn("| Concurrency 4 clients (aggregate tok/s) |", compare.compare([self.data / "result.json"]))

    def test_a_resumed_round_answers_its_levels_from_the_journal(self):
        self.server.step = 0.0005
        protocol = self.protocol
        options = dict(parts=("speed", "concurrency"), warmups=0, repetitions=1, log=lambda m: None,
                       context_tokens=4096, url=self.url + "/", stage="full")
        loads = lambda path: protocol if Path(path).name == "protocol.json" else json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
        with patch.object(runner, "load_json", loads):
            first = runner.run(None, "unused-model", self.data, self.data / "a.json", **options)
            served = self.server.peak
            self.server.shutdown()   # nothing left to serve: every request is in the journal
            again = runner.run(None, "unused-model", self.data, self.data / "b.json",
                               resume=self.data / "a.journal.jsonl", **options)
        self.assertEqual(served, 4)
        self.assertEqual(again["concurrency"]["levels"], first["concurrency"]["levels"])
        self.assertEqual(again["journal"]["live_calls"], 0)

    def test_batched_ceiling_terms(self):
        self.assertAlmostEqual(batched.touched(64, 8, 1), 0.125)
        self.assertAlmostEqual(batched.touched(64, 8, 8), 1 - (1 - 0.125) ** 8)
        self.assertEqual(batched.touched(0, 0, 8), 0.0)
        dense = {"resident_bytes": 10e9, "expert_bytes": 0, "experts": 0, "experts_per_token": 0,
                 "dense_flops": {"bf16": 2 * 5e9}, "expert_flops": {}, "kv_bytes_per_token": 100_000,
                 "fixed_state_bytes": 0, "attention_flops_per_pair": 0}
        one, _ = batched.step_floor(dense, 1, 1000, 1000.0, {"bf16": 200.0})
        many, _ = batched.step_floor(dense, 64, 1000, 1000.0, {"bf16": 200.0})
        self.assertEqual((one["bound"], round(one["memory_ms"], 3)), ("memory", 10.1))
        # 64 rows read the weights once and their own KV: the step costs far less than 64 single steps.
        self.assertLess(many["step_ms"], 64 * one["step_ms"] / 8)
        self.assertAlmostEqual(many["ceiling_tps"], 64 / many["step_ms"] * 1000)
        # At a short context the rows' KV is small and enough rows make the step compute bound.
        self.assertEqual(batched.step_floor(dense, 1024, 10, 1000.0, {"bf16": 200.0})[0]["bound"], "compute")
        moe = {**dense, "resident_bytes": 2e9, "expert_bytes": 64e9, "experts": 64, "experts_per_token": 8,
               "expert_flops": {"bf16": 2 * 64e9 / 2}}
        a, _ = batched.step_floor(moe, 1, 1000, 1000.0, {"bf16": 200.0})
        b, _ = batched.step_floor(moe, 16, 1000, 1000.0, {"bf16": 200.0})
        self.assertAlmostEqual(a["expert_share"], 0.125)
        self.assertGreater(b["memory_ms"], a["memory_ms"] * 4)   # 16 rows touch most of the experts
        self.assertEqual(batched.step_floor(None, 1, 10, 1000.0, {"bf16": 1.0}), (None, ["model weight headers"]))
        self.assertEqual(batched.step_floor(dense, 1, 10, 1000.0, {"fp8": 1.0})[1], ["bf16 GEMM rate"])

    def test_percentiles_and_reply_comparison(self):
        self.assertEqual(concurrency.percentile([5, 1, 3, 2, 4], .5), 3)
        self.assertEqual(concurrency.percentile([5, 1, 3, 2, 4], .95), 5)
        self.assertIsNone(concurrency.percentile([], .5))
        self.assertEqual(concurrency.same_reply([1, 2, 3], [1, 2, 3, 4], None), (True, None))
        self.assertEqual(concurrency.same_reply([1, 9, 3], [1, 2, 3, 4], None), (False, 1))
        # A shorter reply whose last word encoded into other tokens is the same text.
        strings = ["", "ab", "c", "abc", "x"]
        self.assertEqual(concurrency.same_reply([4, 3], [4, 1, 2, 4], strings), (True, None))


if __name__ == "__main__":
    unittest.main()
