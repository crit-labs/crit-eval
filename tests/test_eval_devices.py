"""Several GPUs (protocol context.devices): a probe of every device and the links between them, their memory pooled
for the context math, the decode floor of weights split across them, and device memory sampled on every member.
Host only: the machine these were written on has one GPU."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from crit_eval.limits import context, devices, probe  # noqa: E402
from crit_eval.run import host, runner  # noqa: E402
from test_eval import DENSE, RESIDENT, SHAPES, safetensors_model  # noqa: E402

GB = 10**9


def member(index, bandwidth=1400.0, free=30 * GB, total=32 * GB, tflops=None):
    return {"index": index, "device": f"GPU {index}", "bandwidth_gbs": bandwidth, "memory_free_bytes": free,
            "memory_total_bytes": total, "tflops": tflops or {"bf16": 200.0, "fp8": 400.0}}


def pair(link=25.0, **kwargs):
    first, second = member(0, **kwargs), member(1, **kwargs)
    return {**first, "devices": [first, second],
            "links": [{"from": 0, "to": 1, "gbs": link, "peer_access": True},
                      {"from": 1, "to": 0, "gbs": link * 1.2, "peer_access": True}]}


class DeviceSetTests(unittest.TestCase):
    def test_members_and_their_pooled_memory(self):
        single = member(0)
        self.assertEqual(devices.members(single), [single])
        self.assertIsNone(devices.indices(single))
        self.assertIs(devices.pooled(single, GB, 3 * GB), single)
        two = pair()
        self.assertEqual(devices.indices(two), [0, 1])
        pooled = devices.pooled(two, GB, 3 * GB)
        # max_context takes one member's headroom and reserve off the pool; the other's comes off here.
        self.assertEqual((pooled["memory_free_bytes"], pooled["memory_total_bytes"]), (56 * GB, 64 * GB))
        self.assertEqual(devices.link_gbs(two, 1, 0), 30.0)

    def test_the_split_floor_is_the_faster_of_tensor_and_pipeline(self):
        two = pair(link=25.0)
        split = devices.split_floor(16 * GB, two, layers=64, hidden=5120)
        reads = 16 / 2800 * 1000
        all_reduce = 2 * (1 / 2) * 5120 * 2 / 25e9 * 1000
        self.assertAlmostEqual(split["tensor_ms"], reads + 2 * 64 * all_reduce)
        self.assertAlmostEqual(split["pipeline_ms"], 2 * reads + 5120 * 2 / 25e9 * 1000)
        self.assertEqual(split["split"], "tensor")
        self.assertAlmostEqual(split["bandwidth_gbs"], 16 / (split["floor_ms"] / 1000))
        # A slow link makes the all-reduces cost more than reading twice: pipeline.
        slow = devices.split_floor(16 * GB, pair(link=0.001), layers=64, hidden=5120)
        self.assertEqual(slow["split"], "pipeline")
        self.assertIsNone(devices.split_floor(16 * GB, member(0), 64, 5120))
        unlinked = {**pair(), "links": []}
        self.assertIn("the links between members (crit eval probe --devices)",
                      devices.split_floor(16 * GB, unlinked, 64, 5120)["missing"])

    def test_a_set_is_priced_as_one_device_for_the_floors(self):
        two = pair(tflops={"bf16": 200.0, "fp8": 400.0})
        two["devices"][1]["tflops"] = {"bf16": 100.0}
        found = devices.combined(two, 16 * GB, 64, 5120)
        self.assertAlmostEqual(found["bandwidth_gbs"], devices.split_floor(16 * GB, two, 64, 5120)["bandwidth_gbs"])
        self.assertEqual(found["tflops"], {"bf16": 300.0}, "a format every member measured, summed")
        self.assertEqual([m["index"] for m in found["device_set"]["members"]], [0, 1])
        single = member(0)
        self.assertIs(devices.combined(single, 16 * GB, 64, 5120), single)

    def test_max_context_pools_the_members_memory(self):
        settings = runner.load_json(runner.EVAL_DIR / "protocol.json")["context"]
        overhead = context.GIB + int(settings["reserve_gib"] * context.GIB)
        with tempfile.TemporaryDirectory() as directory:
            model = safetensors_model(directory, SHAPES, DENSE)
            # One device holds the weights and 1,024 tokens of KV; two of them hold the weights once and the rest as KV.
            one = {"device": "fixture", "memory_free_bytes": overhead + RESIDENT + 32 * 1024}
            alone = context.max_context(model, one, context.GIB, settings)
            both = {**one, "devices": [{**one, "index": 0}, {**one, "index": 1}]}
            pooled = context.max_context(model, both, context.GIB, settings)
            self.assertEqual(alone["tokens"], 1024)
            self.assertEqual(pooled["tokens"], (32 * 1024 * 2 + RESIDENT) // 32 // 1024 * 1024)
            self.assertGreater(pooled["tokens"], alone["tokens"])

    def test_device_memory_sums_every_member(self):
        rows = [["0", "1000", "31000", "32000"], ["1", "3000", "29000", "32000"]]
        with patch.object(host, "_nvidia_smi", return_value="nvidia-smi"), patch.object(host, "_query", return_value=rows):
            both = host.device_memory(indices=[0, 1])
            self.assertEqual(both["free_bytes"], 4000 * host.MIB)
            self.assertEqual([d["index"] for d in both["devices"]], [0, 1])
            self.assertEqual(host.device_memory(index=1)["free_bytes"], 3000 * host.MIB)
            self.assertIsNone(host.device_memory(indices=[0, 2]))

    def test_the_probe_runs_every_device_asked_for_and_refuses_others(self):
        with patch.object(probe, "run_binary", return_value={"devices": 2}):
            self.assertEqual(probe.device_indices("binary", None), [0])
            self.assertEqual(probe.device_indices("binary", "all"), [0, 1])
            self.assertEqual(probe.device_indices("binary", [1]), [1])
            for bad in ([2], [0, 0], []):
                with self.subTest(bad=bad), self.assertRaises(SystemExit):
                    probe.device_indices("binary", bad)


if __name__ == "__main__":
    unittest.main()
