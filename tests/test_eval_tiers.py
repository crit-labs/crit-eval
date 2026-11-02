"""Memory tiers (protocol headroom.tiers): the weights a placement leaves off the device priced at host memory's
tier for an engine that declares no host bytes, routed experts by the share a token activates at the device cache's
hit rate (the engine's own, else even routing), and the same tiers in the prefill, first-token and curve floors."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from crit_eval.limits import ceiling as model_ceiling, context, headroom, tiers  # noqa: E402
from crit_eval.run import runner  # noqa: E402
from test_eval import DENSE, RESIDENT, SHAPES, safetensors_model  # noqa: E402

GB = 1e9
# 64 GB of experts, 16 of them on the device; a token routes 1/16 of them (4 GB); rates as the RTX 5090 PC's.
WHERE = {"scenario": "sparse_device_host", "tokens": 8192,
         "bytes": {"resident": 10 * 10**9, "resident_off_device": 0, "experts": 64 * 10**9,
                   "experts_on_device": 16 * 10**9, "experts_off_device": 48 * 10**9},
         "active_expert_bytes_per_token": 4 * 10**9,
         "rates_gbs": {"device": 1400, "link": 27, "host_memory_read": 45}, "cpu_gflops": 1185}
CEILING = {"matmul_flops_per_token": {"nvfp4": 8 * 10**9}, "weight_bytes": 74 * 10**9, "kv_bytes_per_token": 0,
           "attention_flops_per_pair": 0, "recurrent_flops_per_token": 0}
RATES = {"bf16": 200.0, "nvfp4": 800.0}


def sample(hits=None, reads=None, **more):
    return {"warmup": False, **({"expert_hits": hits, "expert_reads": reads} if reads else {}), **more}


class TierTests(unittest.TestCase):
    def test_a_placement_gives_the_tiers_and_a_pass_its_missed_experts_on_the_faster_path(self):
        found = tiers.derive(WHERE, CEILING)
        self.assertEqual((found["source"], found["active_share"], found["even_hit_rate"]),
                         ("placement", 4 / 64, 0.25))
        self.assertIsNone(tiers.derive({**WHERE, "scenario": "fits_device",
                                        "bytes": {**WHERE["bytes"], "experts_off_device": 0}}, CEILING))
        even = tiers.decode_host(found)
        # Even routing: a token's experts hit the device cache as often as it holds them, 25%.
        self.assertEqual((even["cpu"], even["link"], even["hit_rate_source"]), (3 * 10**9, 0, "even routing"))
        engine = tiers.decode_host(found, 0.9)   # S-MBU: only the activated experts, at the engine's hit rate
        self.assertAlmostEqual(engine["cpu"], 0.4 * 10**9, delta=1)
        self.assertEqual((engine["hit_rate"], engine["hit_rate_source"]), (0.9, "engine"))
        # Host memory reads faster than the link carries: a pass's misses are computed on the CPU, unless the link wins.
        faster_link = tiers.decode_host(tiers.derive({**WHERE, "rates_gbs": {**WHERE["rates_gbs"], "link": 60}}, CEILING))
        self.assertEqual((faster_link["cpu"], faster_link["link"]), (0, 3 * 10**9))
        self.assertEqual(tiers.hit_rate([sample(90, 100), sample(10, 100), {"warmup": False}]), 0.5)
        self.assertIsNone(tiers.hit_rate([{"warmup": False}]))

    def test_a_prompt_moves_each_off_device_expert_it_touches_on_the_faster_path(self):
        found = tiers.derive(WHERE, CEILING)
        one = tiers.prompt_host(found, 1)
        self.assertAlmostEqual(one["host_bytes"], 3 * 10**9, delta=1)   # one token: its missed experts
        self.assertEqual(one["path"], "cpu")
        long = tiers.prompt_host(found, 8192)
        # 8,192 tokens touch every expert: the 48 GB off the device cross once, over the link (the CPU's arithmetic on
        # them for every token would take far longer).
        self.assertAlmostEqual(long["host_bytes"], 48 * 10**9, delta=10**6)
        self.assertEqual(long["path"], "link")
        self.assertAlmostEqual(long["host_ms"], 48 / 27 * 1000, places=0)
        self.assertGreater(long["paths_ms"]["cpu"], long["paths_ms"]["link"])

    def test_the_prompt_floors_take_the_host_tier(self):
        plain, _ = model_ceiling.prefill_floor_ms(CEILING, RATES, 8192)
        tiered, _ = model_ceiling.prefill_floor_ms({**CEILING, "tiers": tiers.derive(WHERE, CEILING)}, RATES, 8192)
        self.assertNotIn("host", plain)
        self.assertEqual(tiered["bound"], "host")
        self.assertAlmostEqual(tiered["compute_ms"], plain["total_ms"])
        self.assertAlmostEqual(tiered["total_ms"], 48 / 27 * 1000, places=0)
        ttft, _ = model_ceiling.ttft_floor_ms({**CEILING, "tiers": tiers.derive(WHERE, CEILING)}, RATES, 1400, 8192)
        self.assertEqual(ttft["bound"], "host")
        # The weights on the device are read at its bandwidth; those off it are the host's.
        self.assertAlmostEqual(ttft["memory_ms"], 26 / 1400 * 1000, places=1)

    def test_an_engine_without_host_bytes_is_priced_per_case_at_its_own_hit_rate(self):
        speed = {"decode_score": 10.0, "cases": {
            "chat": {"status": "measured", "decode_tps": 12.0, "round_ms": 80.0, "prompt_tokens": 100,
                     "output_tokens": 256, "samples": [sample(90, 100)]},
            "code": {"status": "measured", "decode_tps": 8.0, "round_ms": 120.0, "prompt_tokens": 100,
                     "output_tokens": 256, "samples": [sample(50, 100)]}}}
        info = {"bytes_per_pass": 14 * 10**9}
        found = tiers.derive(WHERE, CEILING)
        room = headroom.summarize(info, speed, 100.0, 1400, {}, ceiling=CEILING, tflops=RATES,
                                  host_rates={"link": 27, "memory": 45}, tiers=found)
        chat, code = room["cases"]["chat"], room["cases"]["code"]
        self.assertEqual((chat["expert_hit_rate"], code["expert_hit_rate"]), (0.9, 0.5))
        # The longer of the device's bytes at its bandwidth and the misses at host memory's read rate.
        self.assertAlmostEqual(chat["pass_floor_ms"], max(13.6 / 1400, 0.4 / 45) * 1000, places=3)   # the device's
        self.assertAlmostEqual(code["pass_floor_ms"], max(12.0 / 1400, 2.0 / 45) * 1000, places=3)   # host memory's
        self.assertEqual(room["host_pass"]["hit_rate"], 0.7)   # the run's: 140 of 200 reads
        self.assertEqual(room["tiers"]["engine_hit_rate"], 0.7)
        # A declaration keeps the engine's own path.
        declared = headroom.summarize({**info, "host_bytes_per_pass": {"link": 10**9}}, speed, 100.0, 1400, {},
                                      host_rates={"link": 27, "memory": 45}, tiers=found)
        self.assertAlmostEqual(declared["cases"]["chat"]["pass_floor_ms"], 1 / 27 * 1000, places=3)
        self.assertNotIn("expert_hit_rate", declared["cases"]["chat"])
        # Without tiers, every byte stays at the device's bandwidth.
        alone = headroom.summarize(info, speed, 100.0, 1400, {})
        self.assertAlmostEqual(alone["cases"]["chat"]["pass_floor_ms"], 14 / 1400 * 1000 + 0, places=3)

    def test_the_curve_prices_each_point_at_its_hit_rate_and_its_first_token_at_the_host_tier(self):
        points = {"32768": {"status": "measured", "context": 32768, "filled_context": 30000, "prompt_tokens": 29744,
                            "output_tokens": 256, "decode_tps": 5.0, "ttft_ms": 9000.0, "round_ms": 200.0,
                            "tokens_per_round": 1.0, "samples": [sample(60, 100)]}}
        found = {**tiers.derive(WHERE, CEILING), "engine_hit_rate": None}
        info = {"bytes_per_pass": 14 * 10**9, "kv_bytes_per_token": 0}
        summary = headroom.context_summary(info, {"points": points}, {**CEILING, "tiers": found}, RATES, 1400, None, 0,
                                           {}, priced=lambda rows: tiers.decode_host(found, tiers.hit_rate(
                                               [s for r in rows for s in r["samples"]])))
        row = summary["points"]["32768"]
        self.assertAlmostEqual(row["round_floor_ms"], 1.6 / 45 * 1000, places=3)
        self.assertEqual(row["ttft_floor"]["bound"], "host")
        self.assertEqual(row["filled_context"], 30000)


class PlacementTests(unittest.TestCase):
    def test_weights_read_on_every_pass_that_do_not_fit_are_the_host_tiers_bytes(self):
        settings = {**runner.load_json(runner.EVAL_DIR / "protocol.json")["context"], "reserve_gib": 0}
        with tempfile.TemporaryDirectory() as directory:
            model = safetensors_model(directory, SHAPES, DENSE)
            # The device holds all but 1,000 bytes of the resident weights, and no KV: 16 tokens' KV (32 bytes each)
            # pushes as many more off it.
            device = {"bandwidth_gbs": 1000, "host_memory_gbs": {"read": 50}, "host_memory_total_bytes": 10**10,
                      "host_to_device_gbs": {"pinned": {"268435456": 25}},
                      "memory_free_bytes": context.GIB + RESIDENT - 1000}
            found = context.placement(model, device, context.GIB, 2 * context.GIB, settings, 16)
            self.assertEqual(found["scenario"], "dense_device_host")
            self.assertEqual(found["bytes"]["resident_off_device"], 1000 + 32 * 16)
            self.assertIn("every pass reads them from host memory", context.render_placement(found))
            derived = tiers.derive(found)
            self.assertEqual(tiers.decode_host(derived)["cpu"], 1000 + 32 * 16)
            # A requirement beyond the context the device holds next to every weight: the overflow is the host's too.
            short = {**device, "memory_free_bytes": context.GIB + RESIDENT + 32 * 4096}
            found = context.placement(model, short, context.GIB, 2 * context.GIB, settings, 8192)
            self.assertEqual(found["bytes"]["resident_off_device"], 32 * 4096)
            roomy = {**device, "memory_free_bytes": device["memory_free_bytes"] + 10**6}
            self.assertEqual(context.placement(model, roomy, context.GIB, 2 * context.GIB, settings, 8192)["bytes"]
                             ["resident_off_device"], 0)


if __name__ == "__main__":
    unittest.main()
