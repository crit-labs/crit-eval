"""Qualifying the yardstick for a model: the 16-bit baseline (protocol quality.baseline) and the canaries (protocol
quality.canaries), measured by the reference (crit_eval.quality.reference: StreamingReference dtype and canary,
store_baseline, store_canaries) on tiny checkpoints written here, and reported, never judged, by every run
(crit_eval.run.quality_checks.qualify) and its status. Needs torch, transformers and safetensors for the reference
parts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crit_eval import EVAL_DIR, load_json  # noqa: E402
from crit_eval.quality import fixed  # noqa: E402
from crit_eval.reports import metrics  # noqa: E402

HAS_REFERENCE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))
PROTOCOL = load_json(EVAL_DIR / "protocol.json")


def tiny_checkpoint(folder, dtype="bfloat16", initializer_range=0.2):
    """A llama-style checkpoint (rope, grouped attention) of four layers with FP32 weights and `dtype` as its config's
    dtype; weights wide enough (initializer_range) for its outputs to be far from uniform."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False,
        initializer_range=initializer_range)).float().eval().save_pretrained(folder)
    config = json.loads((Path(folder) / "config.json").read_text())
    config.pop("dtype", None)
    config.pop("torch_dtype", None)
    if dtype is not None:
        config["dtype"] = dtype
    (Path(folder) / "config.json").write_text(json.dumps(config))
    return folder


def texts():
    import torch
    torch.manual_seed(3)
    return {"chat": (torch.randint(2, 128, (90,)).tolist(), False), "long@200": (torch.randint(2, 128, (200,)).tolist(), True)}


LAYOUT = {"chat": ("conversations", 40), "long@200": ("long-200", 150)}
# Bootstrap draws small enough for a CPU test (protocol quality.position_spread draws 200 of 64-position blocks).
RULE = {**PROTOCOL["quality"], "position_spread": {"block": 8, "draws": 20, "seed": 0}}


def rows_of(scored):
    """Rows of score_texts in the stored form (fixed.load) and in an engine's (text_quality's two sides)."""
    target, top, values = scored
    return ([{"logprob": lp, "top_ids": ids, "top_logprobs": vals}
             for lp, ids, vals in zip(target.tolist(), top.tolist(), values.tolist())],
            [{"logprob": lp, "top": [[int(i), float(v)] for i, v in zip(ids, vals)]}
             for lp, ids, vals in zip(target.tolist(), top.tolist(), values.tolist())])


@unittest.skipUnless(HAS_REFERENCE, "needs torch, transformers and safetensors")
class ReferencePassTests(unittest.TestCase):
    def score(self, folder, **options):
        from crit_eval.quality import reference
        ref = reference.StreamingReference(folder, "cpu", **options)
        found = texts()
        with tempfile.TemporaryDirectory() as scratch:
            rows = ref.score_texts([ids for ids, _ in found.values()], 8, None, Path(scratch) / "s", log=lambda m: None,
                                   long={1})
        return rows, ref

    def drift(self, exact, other):
        """text_quality of `other`'s rows against `exact`'s, per text."""
        return [metrics.text_quality(rows_of(a)[0], rows_of(b)[1]) for a, b in zip(exact, other)]

    def test_the_16_bit_pass_differs_from_the_fp32_scores_but_by_little(self):
        # The baseline's arithmetic (StreamingReference dtype): the checkpoint's 16-bit dtype through the official
        # forward, the head's product accumulated in FP32. On a tiny model it moves the scores, but by little.
        import torch
        with tempfile.TemporaryDirectory() as folder:
            tiny_checkpoint(folder)
            exact, ref = self.score(folder)
            sixteen, ref16 = self.score(folder, dtype="bfloat16")
            again, _ = self.score(folder, dtype="bfloat16")
            self.assertNotIn("dtype", ref.identity("m"))
            self.assertEqual(ref16.identity("m")["dtype"], "bfloat16")
            self.assertEqual(ref16.skeleton.layers[0].self_attn.q_proj.weight.dtype, torch.bfloat16)
        for a, b in zip(sixteen, again):   # repeatable
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        for a, b in zip(exact, sixteen):
            self.assertEqual(a[0].dtype, torch.float64)
            self.assertFalse(torch.equal(a[0], b[0]))
        for row in self.drift(exact, sixteen):
            self.assertGreater(row["kld_mean"], 0)
            self.assertLess(row["kld_mean"], 0.01)
            self.assertGreater(row["top1_agreement"], 0.95)

    def test_each_canary_moves_the_scores_clearly(self):
        # protocol quality.canaries: each broken forward lands far further from the exact scores than the 16-bit pass.
        from crit_eval.quality import reference
        with tempfile.TemporaryDirectory() as folder:
            tiny_checkpoint(folder)
            exact, _ = self.score(folder)
            sixteen = max(row["kld_mean"] for row in self.drift(exact, self.score(folder, dtype="bfloat16")[0]))
            for name in reference.CANARIES:
                broken, ref = self.score(folder, canary=name)
                self.assertIs(ref.canary_applies, True, name)
                self.assertEqual(ref.identity("m")["canary"], name)
                for row in self.drift(exact, broken):
                    self.assertGreater(row["kld_mean"], max(0.02, 50 * sixteen), name)
            with self.assertRaises(ValueError):
                reference.StreamingReference(folder, "cpu", canary="other")
            with self.assertRaises(ValueError):
                reference.StreamingReference(folder, "cpu", dtype="float32")
            self.assertEqual(reference.StreamingReference(folder, "cpu", canary="skip_layer").skipped, 2)

    def test_rope_base_builds_a_scaled_ropes_table_from_the_doubled_base_by_its_own_rule(self):
        # A rotary module is built again from its config with rope_theta doubled: for the standard rope that is the
        # scaled table, for yarn the table yarn makes of the doubled base, which scaling the built table is not.
        import torch
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from crit_eval.quality import reference
        plain = Qwen3Config(hidden_size=64, num_attention_heads=4, head_dim=16, max_position_embeddings=4096)
        built = Qwen3RotaryEmbedding(plain)
        again = reference._rebuilt_rotary(built)
        n = built.inv_freq.shape[-1]
        scaled = built.inv_freq.double() * 2.0 ** (-torch.arange(n, dtype=torch.float64) / n)
        torch.testing.assert_close(again.inv_freq.double(), scaled, rtol=1e-6, atol=0)
        self.assertEqual(plain.rope_parameters["rope_theta"], built.config.rope_parameters["rope_theta"])   # a copy
        yarn = Qwen3Config(hidden_size=256, num_attention_heads=4, head_dim=64, max_position_embeddings=16384,
                           rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                                            "original_max_position_embeddings": 4096})
        built = Qwen3RotaryEmbedding(yarn)
        again = reference._rebuilt_rotary(built)
        doubled = reference._doubled_theta(yarn)
        self.assertEqual(doubled.rope_parameters["rope_theta"], 20000.0)
        want, scaling = ROPE_INIT_FUNCTIONS["yarn"](doubled)
        torch.testing.assert_close(again.inv_freq, want)
        n = built.inv_freq.shape[-1]
        naive = built.inv_freq.double() * 2.0 ** (-torch.arange(n, dtype=torch.float64) / n)
        self.assertFalse(torch.allclose(again.inv_freq.double(), naive, rtol=1e-4))
        self.assertEqual(again.attention_scaling, scaling)

    def test_kv_shift_moves_an_own_kernels_keys_and_values_and_leaves_interface_attention_to_its_shift(self):
        import torch
        from crit_eval.quality import reference

        class ToyAttention(torch.nn.Module):   # calls its own kernel, never transformers' interface
            def __init__(self):
                super().__init__()
                self.k_proj, self.v_proj = torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return torch.softmax(x @ self.k_proj(x).transpose(-1, -2), -1) @ self.v_proj(x)
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        model = torch.nn.ModuleDict({"own": ToyAttention(),
                                     "interface": Qwen3Attention(Qwen3Config(hidden_size=64, num_attention_heads=4,
                                                                             num_key_value_heads=2, head_dim=16), 0)})
        holder = type("Reference", (), {"canary_applies": None})()
        x = torch.randn(1, 5, 4)
        keys = model["own"].k_proj(x)
        self.assertEqual(reference._own_attention_shift(model, holder), ["own"])
        self.assertIsNone(holder.canary_applies)
        shifted = model["own"].k_proj(x)
        torch.testing.assert_close(shifted[:, 1:], keys[:, :-1])
        torch.testing.assert_close(shifted[:, :1], keys[:, :1])
        self.assertTrue(holder.canary_applies)

    def test_coarse_weights_rounds_the_head_and_the_embedding_rows_too(self):
        import torch
        from crit_eval.quality import reference
        table = torch.randn(6, 64)
        rows = torch.tensor([4, 1])
        torch.testing.assert_close(reference.coarse_grid(table[rows]), reference.coarse_grid(table)[rows])

    def test_the_canary_hooks_name_no_model(self):
        # Each canary is a general hook on the official forward: the coarse grid, the doubled rope base and the
        # shifted keys and values, checked on plain tensors.
        import torch
        from crit_eval.quality import reference
        weights = torch.randn(3, 70)
        grid = reference.coarse_grid(weights)
        self.assertEqual((grid.shape, grid.dtype), (weights.shape, weights.dtype))
        for row in range(3):
            for start in range(0, 70, 32):
                group = weights[row, start:start + 32]
                scale = group.abs().max() / 3
                self.assertLessEqual(len(set((grid[row, start:start + 32] / scale).round().tolist())), 7)
        self.assertTrue(torch.equal(reference.coarse_grid(torch.zeros(2, 4)), torch.zeros(2, 4)))
        theta, d = 10000.0, 16
        standard = 1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
        doubled = 1.0 / (2 * theta) ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
        n = standard.shape[0]
        self.assertTrue(torch.allclose(standard * 2.0 ** (-torch.arange(n, dtype=torch.float64) / n), doubled))
        seen = []

        def inner(module, query, key, value, *args, **kwargs):
            seen.append((key.clone(), value.clone()))
            return query, None
        key = torch.arange(5.0).reshape(1, 1, 5, 1)
        with patch("transformers.AttentionInterface._global_mapping", {"sdpa": inner}):
            reference._shifted_attention("sdpa")(None, key, key, key + 10)
        self.assertEqual(seen[0][0].flatten().tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(seen[0][1].flatten().tolist(), [10, 10, 11, 12, 13])

    def test_the_baseline_is_stored_repeatable_in_the_configs_dtype(self):
        # protocol quality.baseline: one pass in the config's 16-bit dtype, per set and path its row and bootstrap
        # resamples, stored beside the scores; stored once, again the same bytes, and none for a config without one.
        from crit_eval.quality import reference
        found = texts()
        with tempfile.TemporaryDirectory() as temp:
            folder = tiny_checkpoint(Path(temp) / "bf16")
            exact = reference.StreamingReference(folder, "cpu")
            identity = {**exact.identity("m"), **fixed.identity(found, 4, None)}
            datas = []
            for name in ("one", "two"):
                data = Path(temp) / name
                reference.store_texts(exact, found, 4, None, data, "m", log=lambda m: None)
                logged = []
                how = reference.store_baseline(folder, data, found, 4, None, identity, LAYOUT, RULE, log=logged.append,
                                               device="cpu")
                self.assertEqual(how, "measured")
                datas.append(data)
            one, two = (fixed.load_beside(data, fixed.BASELINE, identity) for data in datas)
            self.assertEqual(one["sets"], two["sets"])   # repeatable
            self.assertEqual(one["dtype"], "bfloat16")
            self.assertEqual(sorted(one["sets"]), ["conversations/prompt", "conversations/served", "long-200/prompt",
                                                   "long-200/served"])
            prompt = one["sets"]["long-200/prompt"]
            self.assertEqual((prompt["targets"], len(prompt["bootstrap"])), (149, 20))
            self.assertGreater(prompt["kld_mean"], 0)
            stored = json.loads((datas[0] / fixed.STORE / fixed.BASELINE).read_text())["identity"]
            self.assertEqual((stored["baseline"], stored["bootstrap"]), ("bfloat16", {"block": 8, "draws": 20, "seed": 0}))
            self.assertTrue(any(line.startswith("reference baseline long-200/prompt") for line in logged), logged)
            # What it holds: the 16-bit pass's rows against the stored scores, as an engine's would be.
            sixteen = reference.StreamingReference(folder, "cpu", dtype="bfloat16")
            rows = dict(zip(found, sixteen.score_texts([ids for ids, _ in found.values()], 4, None,
                                                       Path(temp) / "scratch", log=lambda m: None, long={1})))
            again = rows_of(rows["chat"])[1]
            scores = fixed.load(datas[0], identity)
            self.assertEqual({k: v for k, v in one["sets"]["conversations/served"].items() if k != "bootstrap"},
                             metrics.text_quality(scores["chat"][39:], again[39:]))
            self.assertEqual(one["sets"]["conversations/served"]["bootstrap"],
                             metrics.bootstrap_samples(scores["chat"][39:], again[39:], 8, 20, 0))
            self.assertEqual(reference.store_baseline(folder, datas[0], found, 4, None, identity, LAYOUT, RULE,
                                                      log=lambda m: None, device="cpu"), "stored")
            # Another bootstrap rule is another identity: measured again.
            other = {**RULE, "position_spread": {"block": 8, "draws": 10, "seed": 0}}
            self.assertEqual(reference.store_baseline(folder, datas[1], found, 4, None, identity, LAYOUT, other,
                                                      log=lambda m: None, device="cpu"), "measured")
            # The config's dtype decides: float16, and none for an FP32 config (an empty baseline that says so).
            half = tiny_checkpoint(Path(temp) / "fp16", dtype="float16")
            self.assertEqual(reference.sixteen_bit_dtype(half), "float16")
            single = tiny_checkpoint(Path(temp) / "fp32", dtype="float32")
            self.assertIsNone(reference.sixteen_bit_dtype(single))
            # A config that declares none: the stored tensors decide (these store FP32 only: none)
            undeclared = tiny_checkpoint(Path(temp) / "undeclared", dtype=None)
            self.assertIsNone(reference.sixteen_bit_dtype(undeclared))
            header = lambda dtype, *shape: type("Header", (), {"dtype": dtype, "shape": shape})()
            self.assertEqual(reference.stored_sixteen_bit([header("BF16", 4, 4), header("F32", 4)]), "bfloat16")
            self.assertEqual(reference.stored_sixteen_bit([header("F16", 4, 4), header("BF16", 4)]), "float16")
            self.assertEqual(reference.stored_sixteen_bit([header("F8_E4M3", 4, 4), header("F32", 4)]), "bfloat16")
            self.assertEqual(reference.store_baseline(single, datas[1], found, 4, None, identity, LAYOUT, RULE,
                                                      log=lambda m: None, device="cpu"), "none")
            empty = fixed.load_beside(datas[1], fixed.BASELINE, identity)
            self.assertEqual((empty["dtype"], empty["sets"]), (None, {}))
            self.assertIn("no 16-bit dtype", empty["reason"])

    def test_the_canaries_are_stored_once_each_with_what_does_not_apply(self):
        # protocol quality.canaries: a pass per canary, stored beside the scores; one with nothing to break in the
        # model is recorded as not applicable, one the forward refuses as unsupported.
        from crit_eval.limits.schemes import SchemeError
        from crit_eval.quality import reference
        found = texts()
        with tempfile.TemporaryDirectory() as temp:
            folder = tiny_checkpoint(Path(temp) / "model")
            exact = reference.StreamingReference(folder, "cpu")
            identity = {**exact.identity("m"), **fixed.identity(found, 4, None)}
            data = Path(temp) / "data"
            reference.store_texts(exact, found, 4, None, data, "m", log=lambda m: None)
            logged = []
            self.assertEqual(reference.store_canaries(folder, data, found, 4, None, identity, LAYOUT, log=logged.append,
                                                      device="cpu"), "measured")
            stored = fixed.load_beside(data, fixed.CANARIES, identity)["canaries"]
            self.assertEqual(list(stored), list(reference.CANARIES))
            for name, canary in stored.items():
                self.assertEqual(canary["status"], "measured", name)
                self.assertEqual(sorted(canary["sets"]), ["conversations/prompt", "conversations/served",
                                                          "long-200/prompt", "long-200/served"])
                self.assertGreater(canary["sets"]["long-200/prompt"]["kld_mean"], 0.02, name)
            self.assertTrue(any(line.startswith("reference canary kv_shift long-200/prompt") for line in logged))
            self.assertEqual(reference.store_canaries(folder, data, found, 4, None, identity, LAYOUT,
                                                      log=lambda m: None, device="cpu"), "stored")
            other = Path(temp) / "other"
            reference.store_texts(exact, found, 4, None, other, "m", log=lambda m: None)
            original = reference.StreamingReference._skip

            def refused(self, index, args, kwargs, out):
                if self.canary == "skip_layer":
                    raise SchemeError("a forward that cannot skip")
                return original(self, index, args, kwargs, out)
            with patch.object(reference.StreamingReference, "_double_rope_base", lambda self: False), \
                    patch.object(reference.StreamingReference, "_skip", refused):
                reference.store_canaries(folder, other, found, 4, None, identity, LAYOUT, log=lambda m: None,
                                         device="cpu", names=("rope_base", "skip_layer"))
            stored = fixed.load_beside(other, fixed.CANARIES, identity)["canaries"]
            self.assertEqual((stored["rope_base"]["status"], stored["rope_base"]["sets"]), ("not applicable", {}))
            self.assertIn("rotary", stored["rope_base"]["reason"])
            self.assertEqual((stored["skip_layer"]["status"], stored["skip_layer"]["reason"]),
                             ("unsupported", "a forward that cannot skip"))


LIMITS = {"kld_mean_max": 0.4, "kld_p99_max": 4.0}
SERVED_LIMITS = {"ppl_ratio_max": 1.4, **LIMITS}


def row(kld_mean, kld_p99, ppl_ratio=1.0, **more):
    return {"targets": 100, "kld_mean": kld_mean, "kld_p99": kld_p99, "ppl_ratio": ppl_ratio, "top1_agreement": 0.99,
            **more}


def scored_quality():
    """A scored quality part as run_quality leaves it: the engine's rows, each set's limits and verdict."""
    sets = {"conversations/prompt": {**row(0.05, 0.8), "limits": dict(LIMITS)},
            "conversations/served": {**row(0.01, 0.1, 1.01), "limits": dict(SERVED_LIMITS)},
            "long-1024/prompt": {**row(0.15, 3.2), "limits": dict(LIMITS)},
            "long-4096/prompt": {**row(0.2, 5.08), "limits": dict(LIMITS)}}
    for s in sets.values():
        s["failures"] = metrics.judge(s, s["limits"])
        s["passed"] = not s["failures"]
        s["budget_used"] = metrics.budget_used(s, s["limits"])
    return {"status": "scored", "evidence": "served", "sets": sets, "honesty": {"passed": None, "reason": "x"},
            "skipped": {}, "passed": all(s["passed"] for s in sets.values())}


def baseline_body():
    """A stored 16-bit baseline as store_baseline writes it: long-4096 over p99 4.0, long-1024 within it but with
    bootstrap draws past it, conversations far inside."""
    return {"dtype": "bfloat16", "seconds": 126.0, "sets": {
        "conversations/prompt": {**row(0.014, 0.16), "bootstrap": [[0.014, 0.1 + i / 1000, 1.0] for i in range(20)]},
        "conversations/served": {**row(0.0001, 0.002, 1.001), "bootstrap": [[0.0001, 0.002, 1.001]] * 20},
        "long-1024/prompt": {**row(0.061, 1.87), "bootstrap": [[0.06, 0.5 + 0.2 * i, 1.0] for i in range(20)]},
        "long-4096/prompt": {**row(0.107, 4.26), "bootstrap": [[0.1, 4.0 + i / 10, 1.0] for i in range(20)]}}}


class RunReportTests(unittest.TestCase):
    def test_a_run_reads_the_baseline_and_canaries_stored_for_its_scores(self):
        # quality_checks.stored_qualification: under the stored scores' identity, the baseline for the bootstrap the
        # protocol draws now; none where the data holds none.
        from crit_eval.run import quality_checks
        identity = {"model_sha256": "m", "reference": 3, "top_k": 16}
        noise = PROTOCOL["quality"]["position_spread"]
        bootstrap = {key: noise[key] for key in ("block", "draws", "seed")}
        with tempfile.TemporaryDirectory() as data:
            self.assertEqual(quality_checks.stored_qualification(data, PROTOCOL, identity),
                             {"baseline": None, "canaries": None})
            fixed.write_beside(data, fixed.BASELINE, {**identity, "baseline": "bfloat16", "bootstrap": bootstrap},
                               baseline_body())
            fixed.write_beside(data, fixed.CANARIES, {**identity, "canaries": ["skip_layer"]}, {"canaries": {}})
            found = quality_checks.stored_qualification(data, PROTOCOL, identity)
            self.assertEqual(found["baseline"], {**baseline_body(), "block": bootstrap["block"]})
            self.assertEqual(found["canaries"], {"canaries": {}})
            other_model = quality_checks.stored_qualification(data, PROTOCOL, {**identity, "model_sha256": "x"})
            self.assertIsNone(other_model["baseline"])
            other = {**PROTOCOL, "quality": {**PROTOCOL["quality"], "position_spread": {**noise, "draws": 7}}}
            self.assertIsNone(quality_checks.stored_qualification(data, other, identity)["baseline"])

    def test_attribution_and_the_precheck_read_the_baselines_budget_per_set(self):
        # quality_checks.baseline_budgets: the budget qualify's rows give the baseline of each scored set, or why
        # there is none; run_attribution carries it beside the full engine's budget (base).
        from crit_eval.reports import status
        from crit_eval.run import quality_checks
        quality = scored_quality()
        found = quality_checks.baseline_budgets(quality["sets"], {"baseline": {**baseline_body(), "block": 64}})
        self.assertEqual(found["status"], "measured")
        self.assertEqual(found["budget_used"]["long-4096/prompt"], 4.26 / 4.0)
        self.assertEqual(found["budget_used"], {name: quality_checks.baseline_row(
            baseline_body()["sets"][name], quality["sets"][name]["limits"], {"block": 64})["budget_used"]
            for name in quality["sets"] if name in baseline_body()["sets"]})
        self.assertEqual(quality_checks.baseline_budgets(quality["sets"], None)["status"], "missing")
        self.assertEqual(quality_checks.baseline_budgets(quality["sets"], {"baseline": {"sets": {}, "reason": "fp32"}}),
                         {"status": "none", "reason": "fp32"})
        attribution = {"status": "measured", "base": {"long-4096/prompt": 0.5}, "baseline": found,
                       "variants": {"exact": {"share": {"long-4096/prompt": 0.25},
                                              "budget_used": {"long-4096/prompt": 0.25}}}}
        line = status._render_attribution(attribution, "long-4096/prompt")[0]
        self.assertIn("full engine long-4096/prompt 50%, 16-bit baseline 106%", line)

    def test_the_quality_terms_go_into_the_engines_spec_with_their_sources(self):
        from crit_eval.run import quality_checks
        mode = {"ppl_ratio_max": 1.4, "kld_mean_max": 0.4, "kld_p99_max": 4.0}
        found = quality_checks.quality_spec(mode, None, "speed")
        self.assertEqual((found["quality_limits"]["source"], found["quality_limits"]["value"]), ("mode", mode))
        self.assertNotIn("quality_allowance", found)
        found = quality_checks.quality_spec({**mode, "kld_mean_max": 0.1}, {"kld_mean_max": 0.1}, "speed",
                                            {"factor": 2.0, "keys": ["kld_p99_max"], "sets": {"long-4096/prompt": {}}})
        self.assertEqual(found["quality_limits"]["source"], "user")
        self.assertIn("the run's budget sets kld_mean_max", found["quality_limits"]["note"])
        self.assertEqual(found["quality_allowance"]["source"], "measured")
        self.assertEqual(found["quality_allowance"]["value"]["factor"], 2.0)

    def test_each_set_lists_the_baseline_beside_the_engine_and_no_verdict_changes(self):
        # protocol quality.baseline: per set and path the baseline's drift and the budget it uses of the limits that
        # set is judged by; unreachable where it exceeds them, near where its bootstrap p95 does. Reported only.
        import copy
        from crit_eval.reports import status
        from crit_eval.run import quality_checks
        quality = scored_quality()
        before = copy.deepcopy(quality)
        gate = status._quality_gate({"quality": copy.deepcopy(quality)})
        quality_checks.qualify(quality, {"baseline": {**baseline_body(), "block": 64}, "canaries": None})
        for name, s in before["sets"].items():   # the engine's rows, limits and verdicts as they were
            self.assertEqual({key: quality["sets"][name][key] for key in s}, s, name)
        self.assertEqual(quality["passed"], before["passed"])
        self.assertEqual(quality["baseline"], {"status": "measured", "dtype": "bfloat16",
                                               "unreachable": ["long-4096/prompt"], "near": ["long-1024/prompt"]})
        far, near = quality["sets"]["long-4096/prompt"]["baseline"], quality["sets"]["long-1024/prompt"]["baseline"]
        self.assertEqual((far["reach"], far["kld_p99"], far["budget_used"]), ("unreachable", 4.26, 4.26 / 4.0))
        self.assertEqual(near["reach"], "near")
        self.assertEqual(near["budget_spread"],
                         metrics.spread_of(baseline_body()["sets"]["long-1024/prompt"]["bootstrap"], LIMITS, 64))
        self.assertEqual(quality["sets"]["conversations/served"]["baseline"]["reach"], "within")
        self.assertEqual(quality["canaries"], {"status": "unmeasured",
                                               "reason": "crit eval reference --canaries measures them"})
        after = status._quality_gate({"quality": quality})
        self.assertEqual(after["state"], gate["state"])
        self.assertTrue(after["detail"].startswith(gate["detail"]))
        self.assertIn("a bfloat16 engine of this model: long-4096/prompt unreachable at 16 bits (the baseline uses 106% "
                      "of its limits); long-1024/prompt near the limit (the baseline's bootstrap p95 uses",
                      after["detail"])
        # A raised limit is the one the baseline is read against: long-4096's p99 raised to 6.0 leaves it within.
        raised = scored_quality()
        raised["sets"]["long-4096/prompt"]["limits"]["kld_p99_max"] = 6.0
        quality_checks.qualify(raised, {"baseline": {**baseline_body(), "block": 64}, "canaries": None})
        self.assertEqual(raised["sets"]["long-4096/prompt"]["baseline"]["reach"], "within")
        self.assertEqual(raised["baseline"]["unreachable"], [])
        # Nothing reported of a part that was not scored.
        self.assertEqual(quality_checks.qualify({"status": "unsupported", "reason": "r"}, None),
                         {"status": "unsupported", "reason": "r"})

    def test_data_without_a_baseline_says_so_and_never_fails(self):
        from crit_eval.reports import status
        from crit_eval.run import quality_checks
        quality = quality_checks.qualify(scored_quality(), {"baseline": None, "canaries": None})
        self.assertEqual(quality["baseline"], {"status": "missing", "reason": quality_checks.BASELINE_MISSING})
        gate = status._quality_gate({"quality": quality})
        self.assertEqual(gate["state"], "fail")   # long-4096's own p99, as before
        self.assertIn("; no 16-bit baseline: the data holds no 16-bit baseline (scored before crit-eval-v0.9.0",
                      gate["detail"])
        passing = scored_quality()
        del passing["sets"]["long-4096/prompt"]
        passing["passed"] = True
        self.assertEqual(status._quality_gate({"quality": quality_checks.qualify(passing, None)})["state"], "pass")
        none = quality_checks.qualify(scored_quality(), {"baseline": {"dtype": None, "sets": {}, "reason": "no dtype"}})
        self.assertEqual(none["baseline"], {"status": "none", "reason": "no dtype"})
        refused = quality_checks.qualify(scored_quality(), {"baseline": {"dtype": "float16", "sets": {},
                                                                          "unsupported": "a form"}})
        self.assertEqual(refused["baseline"], {"status": "unsupported", "reason": "a form"})
        self.assertEqual(quality_checks.qualification_lines(refused)[0], "no 16-bit baseline: a form")
        within = scored_quality()
        body = baseline_body()
        body["sets"] = {name: {**s, "kld_p99": 0.4, "bootstrap": None} for name, s in body["sets"].items()}
        quality_checks.qualify(within, {"baseline": body})
        self.assertEqual(quality_checks.qualification_lines(within)[0],
                         "the bfloat16 baseline uses at most 27% of a limit (long-4096/prompt)")

    def test_the_canaries_the_runs_limits_pass_are_named(self):
        # protocol quality.canaries: each stored canary judged against the limits the run judges each set by; a
        # canary every set of which stays within them passes (the limits are too loose for this model).
        from crit_eval.reports import status
        from crit_eval.run import quality_checks
        canaries = {"canaries": {
            "skip_layer": {"status": "measured", "sets": {name: row(0.01, 0.5) for name in scored_quality()["sets"]}},
            "kv_shift": {"status": "measured", "sets": {"long-4096/prompt": row(0.9, 9.0),
                                                        "conversations/prompt": row(0.01, 0.2)}},
            "rope_base": {"status": "not applicable", "reason": "the model holds no rotary frequencies", "sets": {}},
            "coarse_weights": {"status": "measured", "sets": {"long-32768/prompt": row(2.0, 20.0)}}}}
        quality = quality_checks.qualify(scored_quality(), {"baseline": None, "canaries": canaries})
        found = quality["canaries"]
        self.assertEqual(found["passing"], ["skip_layer"])
        self.assertEqual(found["canaries"]["skip_layer"]["passing_sets"], sorted(scored_quality()["sets"]))
        kv = found["canaries"]["kv_shift"]
        self.assertEqual((kv["passed"], kv["passing_sets"], kv["budget_used"]),
                         (False, ["conversations/prompt"], 9.0 / 4.0))
        self.assertEqual(found["canaries"]["rope_base"], {"status": "not applicable", "passed": None,
                                                          "reason": "the model holds no rotary frequencies"})
        self.assertEqual(found["canaries"]["coarse_weights"], {"status": "no shared set", "passed": None})
        detail = status._quality_gate({"quality": quality})["detail"]
        self.assertIn("canaries the limits pass, so the limits are too loose to tell them from a working engine of "
                      "this model: skip_layer", detail)
        self.assertIn("canaries the limits catch (the largest budget each uses): kv_shift 225%", detail)
        self.assertIn("canaries not judged: rope_base (not applicable), coarse_weights (no shared set)", detail)


if __name__ == "__main__":
    unittest.main()
