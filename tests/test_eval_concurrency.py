"""The concurrency part (protocol concurrency): closed-loop clients against a fake server that runs several requests at
once, the levels it measures, its aggregate, peak, per-request and consistency figures, its budget, journal and status,
and the batched ceiling."""
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
from crit_eval.limits import batched  # noqa: E402
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
        self.wfile.write(json.dumps({"object": "list", "data": [{"id": "fake", **self.server.card}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
                          {"choices": [], "usage": usage}):
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            with self.server.lock:
                self.server.active -= 1


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), BatchingChat)
        self.server.card = {"max_concurrent_requests": 4}
        self.server.lock, self.server.active, self.server.peak = threading.Lock(), 0, 0
        self.server.step, self.server.unstable, self.server.usage = 0.002, False, {}
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
        found = self.measure(inputs=lambda: (terms, 1000.0, {"bf16": 100.0}))
        for clients, level in found["levels"].items():
            floor = level["ceiling"]
            self.assertEqual(floor["rows"], int(clients))
            self.assertAlmostEqual(level["fraction_of_ceiling"], level["aggregate_tps"] / floor["ceiling_tps"])
        unpriced = self.measure(inputs=lambda: (None, None, None))["levels"]["1"]
        self.assertIsNone(unpriced["ceiling"])

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
