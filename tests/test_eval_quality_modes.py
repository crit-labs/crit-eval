"""Quality modes: the checkpoint's own activation quantization (crit_eval.limits.schemes.activations, quantize_activations)
and the as_quantized reference that applies it (crit_eval.quality.reference), on hand-built values and a tiny NVFP4 model
written here (needs torch, transformers, safetensors and tokenizers)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crit_eval.quality import fixed  # noqa: E402

HAS_REFERENCE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors", "tokenizers"))
NVFP4_INPUTS = {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": "local",
                "symmetric": True}
FP8_TOKEN_INPUTS = {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True, "symmetric": True}


class Stored:
    """A reference checkpoint seen as crit_eval.limits.schemes reads one: tensors by key, headers, a quantization config."""

    def __init__(self, tensors, quantization):
        self.tensors, self.quantization = tensors, quantization

    def has(self, key):
        return key in self.tensors

    def raw(self, key):
        return self.tensors[key]

    def header(self, key):
        from crit_eval.quality.checkpoints import TORCH_DTYPES
        tensor = self.tensors[key]
        return type("Header", (), {"dtype": TORCH_DTYPES[str(tensor.dtype).split(".")[-1]], "shape": list(tensor.shape)})


def nvfp4_matrix(weight, global_scale):
    """compressed-tensors' NVFP4 tensors of an FP32 [out, in] weight: E2M1 codes (low nibble first), E4M3 scales per 16
    and the global scale, a divisor."""
    import torch
    from crit_eval.limits.schemes import E2M1, _round_e2m1
    blocks = weight.reshape(weight.shape[0], -1, 16)
    stored = (blocks.abs().amax(-1) / 6 * global_scale).to(torch.float8_e4m3fn)
    step = stored.float()[..., None] / global_scale
    values = _round_e2m1((blocks.abs() / torch.where(step > 0, step, 1)).clamp(max=6))
    codes = torch.tensor(E2M1).reshape(1, -1).eq(values.reshape(-1, 1)).float().argmax(-1).reshape(values.shape)
    codes = (codes + 8 * (blocks < 0).long() * (values > 0).long()).reshape(weight.shape[0], -1)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return {"weight_packed": packed, "weight_scale": stored,
            "weight_global_scale": torch.tensor([global_scale], dtype=torch.float32)}


@unittest.skipUnless(HAS_REFERENCE, "needs torch")
class ActivationTests(unittest.TestCase):
    def test_nvfp4_activations_round_to_the_even_code_with_the_stored_scales(self):
        import torch
        from crit_eval.limits.schemes import quantize_activations
        # Global scale 1, group amax 6: the E4M3 scale is 1 and every value rounds to the E2M1 grid, ties to the even
        # code (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4).
        x = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.75, -2.9, 0.1, 0.0, 1.0, 4.4, -6.0, 0.6]])
        expected = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -1.0, -3.0, 0.0, 0.0, 1.0, 4.0, -6.0, 0.5]
        self.assertEqual(quantize_activations(x, "nvfp4", 1.0)[0].tolist(), expected)
        # A global scale of 1.1 stores the group scale 6 / 6 x 1.1 as the E4M3 1.125: the step is 1.125 / 1.1.
        step = 1.125 / 1.1
        got = quantize_activations(x, "nvfp4", 1.1)[0]
        self.assertAlmostEqual(got[7].item(), 6 * step, places=6)
        self.assertAlmostEqual(got[1].item(), 0.5 * step, places=6)   # 0.75 / step = 0.733: the nearest is 0.5
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            quantize_activations(torch.ones(1, 8), "nvfp4", 1.0)

    def test_fp8_activations_take_one_scale_per_token_or_the_stored_one(self):
        import torch
        from crit_eval.limits.schemes import quantize_activations
        x = torch.tensor([[448.0, 1.0, -3.3], [0.5, 0.0, 0.25]])
        got = quantize_activations(x, "fp8_token", None)
        self.assertEqual(got[0].tolist(), [448.0, 1.0, -3.25])   # step 1: E4M3 has 3 mantissa bits
        self.assertEqual(got[1].tolist(), [0.5, 0.0, 0.25])       # step 0.5 / 448: both exact
        self.assertEqual(quantize_activations(torch.tensor([[1000.0]]), "fp8_tensor", 1.0).tolist(), [[448.0]])

    def test_the_checkpoint_declares_its_input_activations(self):
        import torch
        from crit_eval.limits.schemes import SchemeError, activations
        nvfp4 = {"model.layers.0.mlp.up_proj." + k: v for k, v in nvfp4_matrix(torch.randn(32, 32), 448.0).items()}
        nvfp4["model.layers.0.mlp.up_proj.input_global_scale"] = torch.tensor([224.0])
        fp8 = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 32, dtype=torch.float8_e4m3fn),
               "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(16, 1)}
        plain = {"model.layers.0.self_attn.k_proj.weight": torch.zeros(16, 32)}
        groups = {"group_0": {"targets": ["re:.*self_attn.*"], "weights": {"num_bits": 8, "type": "float"},
                              "input_activations": FP8_TOKEN_INPUTS},
                  "group_1": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float"},
                              "input_activations": NVFP4_INPUTS}}
        stored = Stored({**nvfp4, **fp8, **plain}, {"quant_method": "compressed-tensors", "config_groups": groups})
        self.assertEqual(activations(stored, "model.layers.0.mlp.up_proj"), ("nvfp4", 224.0))
        self.assertEqual(activations(stored, "model.layers.0.self_attn.q_proj"), ("fp8_token", None))
        self.assertIsNone(activations(stored, "model.layers.0.self_attn.k_proj"))   # its weights are not quantized
        weight_only = Stored(nvfp4, {"quant_method": "compressed-tensors", "config_groups": {
            "group_0": {"targets": ["Linear"], "weights": {"num_bits": 4}, "input_activations": None}}})
        self.assertIsNone(activations(weight_only, "model.layers.0.mlp.up_proj"))
        odd = Stored(nvfp4, {"quant_method": "compressed-tensors", "config_groups": {
            "group_0": {"targets": ["Linear"], "weights": {"num_bits": 4}, "input_activations": {"num_bits": 8, "type": "int"}}}})
        with self.assertRaisesRegex(SchemeError, "not a form crit emulates"):
            activations(odd, "model.layers.0.mlp.up_proj")
        # ModelOpt stores a static input scale: NVFP4's global scale is its inverse, FP8's the scale itself.
        modelopt = {"model.layers.0.mlp.up_proj.weight": nvfp4["model.layers.0.mlp.up_proj.weight_packed"],
                    "model.layers.0.mlp.up_proj.weight_scale": nvfp4["model.layers.0.mlp.up_proj.weight_scale"],
                    "model.layers.0.mlp.up_proj.weight_scale_2": torch.tensor([1 / 448.0]),
                    "model.layers.0.mlp.up_proj.input_scale": torch.tensor([0.25])}
        self.assertEqual(activations(Stored(modelopt, {"quant_method": "modelopt"}), "model.layers.0.mlp.up_proj"),
                         ("nvfp4", 4.0))

    def test_as_quantized_judges_by_twice_the_preserve_budget(self):
        # protocol quality.mode_thresholds: 4-bit activation math swings KL far more than exact activations, so
        # as_quantized's budget is twice preserve's; a run's own limits lay over either mode's.
        from crit_eval import load_json, EVAL_DIR
        from crit_eval.run import quality_checks
        protocol = load_json(EVAL_DIR / "protocol.json")
        preserve = quality_checks.quality_thresholds(protocol)
        self.assertEqual(preserve, {"ppl_ratio_max": 1.05, "kld_mean_max": 0.05, "kld_p99_max": 0.5})
        self.assertEqual(quality_checks.quality_thresholds(protocol, mode="preserve"), preserve)
        quantized = quality_checks.quality_thresholds(protocol, mode="as_quantized")
        self.assertEqual(quantized, {"ppl_ratio_max": 1.1, "kld_mean_max": 0.1, "kld_p99_max": 1.0})
        self.assertAlmostEqual(quantized["ppl_ratio_max"] - 1, 2 * (preserve["ppl_ratio_max"] - 1))
        self.assertEqual(quality_checks.quality_thresholds(protocol, {"kld_p99_max": 0.8}, "as_quantized"),
                         {**quantized, "kld_p99_max": 0.8})

    def test_a_run_judges_by_the_spread_of_every_text_the_rule_names(self):
        # protocol quality.spread covers every length (from_tokens 0): a spread stored under an earlier rule, for the
        # long texts alone, is not judged by until the reference scores it again.
        from crit_eval import load_json, EVAL_DIR
        from crit_eval.run import quality_checks
        protocol = load_json(EVAL_DIR / "protocol.json")
        self.assertEqual(protocol["quality"]["spread"]["from_tokens"], 0)
        prepared = {"quality": [{"id": "c", "set": "conversations", "prompt_ids": [1, 2], "continuation_ids": [3]},
                                {"id": "l", "set": "long-1024", "prompt_ids": [1] * 9000, "continuation_ids": [3]}]}
        identity = {"reference": 3, "quality_mode": "as_quantized"}
        store = fixed.STORES["as_quantized"]
        spread = {"conversations/prompt": {"kld_mean": 0.04, "kld_p99": 0.6, "ppl_ratio": 1.02}}
        with tempfile.TemporaryDirectory() as data:
            self.assertIsNone(quality_checks.stored_spread(data, prepared, protocol, identity, store))
            fixed.write_spread(data, {**identity, "spread_perturbation": "tf32", "spread_texts": ["l"]}, spread, store)
            self.assertIsNone(quality_checks.stored_spread(data, prepared, protocol, identity, store))   # the long text alone
            fixed.write_spread(data, {**identity, "spread_perturbation": "tf32", "spread_texts": ["c", "l"]}, spread,
                               store)
            self.assertEqual(quality_checks.stored_spread(data, prepared, protocol, identity, store), spread)
            self.assertIsNone(quality_checks.stored_spread(data, prepared, protocol, identity))   # preserve's store has none

    def test_each_quality_mode_has_its_own_store(self):
        with tempfile.TemporaryDirectory() as data:
            scores = {"t": ([-0.5], [[1]], [[-0.5]])}
            fixed.write(data, {"reference": 3}, scores)
            fixed.write(data, {"reference": 3, "quality_mode": "as_quantized"}, {"t": ([-0.9], [[2]], [[-0.9]])},
                        fixed.STORES["as_quantized"])
            self.assertEqual(fixed.load(data, {"reference": 3})["t"][0]["logprob"], -0.5)
            self.assertEqual(fixed.load(data, {"quality_mode": "as_quantized"}, fixed.STORES["as_quantized"])["t"][0]
                             ["logprob"], -0.9)
            self.assertIsNone(fixed.load(data, {"quality_mode": "as_quantized"}))


@unittest.skipUnless(HAS_REFERENCE, "needs torch, transformers, safetensors and tokenizers")
class AsQuantizedReferenceTests(unittest.TestCase):
    def test_the_reference_quantizes_every_declared_input_as_the_checkpoint_does(self):
        import torch
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoModelForCausalLM
        from crit_eval.limits.schemes import activations, quantize_activations
        from test_eval_formats import scored, sequences, tiny
        with tempfile.TemporaryDirectory() as temp:
            original = tiny(Path(temp) / "original")
            model = AutoModelForCausalLM.from_pretrained(original, dtype=torch.float32)
            folder = Path(temp) / "nvfp4"
            folder.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (folder / name).write_bytes((original / name).read_bytes())
            tensors = {}
            for key, value in model.state_dict().items():
                base = key[: -len(".weight")]
                if key.endswith(".weight") and value.dim() == 2 and ".layers." in key:
                    tensors.update({f"{base}.{k}": v for k, v in nvfp4_matrix(value, 448.0 * 6 / 0.2).items()})
                    tensors[f"{base}.input_global_scale"] = torch.tensor([448.0 * 6 / 3.0])
                else:
                    tensors[key] = value.contiguous()
            save_file(tensors, str(folder / "model.safetensors"))
            config = json.loads((original / "config.json").read_text())
            config["quantization_config"] = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                             "config_groups": {"group_0": {"targets": ["Linear"],
                                                                           "weights": {"num_bits": 4, "type": "float",
                                                                                       "group_size": 16},
                                                                           "input_activations": NVFP4_INPUTS}},
                                             "ignore": ["lm_head"]}
            (folder / "config.json").write_text(json.dumps(config))
            exact = scored(folder, sequences())
            with patch_mode("as_quantized"):
                quantized = scored(folder, sequences())
            # The twin: transformers' model with the decoded weights and every declared input quantized alike.
            from crit_eval.quality.checkpoints import _text_config, open_checkpoint
            checkpoint = open_checkpoint(folder, _text_config(folder))
            twin = AutoModelForCausalLM.from_config(AutoConfig.for_model(**{k: v for k, v in config.items()
                                                                          if k != "quantization_config"}),
                                                    dtype=torch.float32, attn_implementation="eager")
            twin.load_state_dict({k: checkpoint.dense(k[: -len(".weight")], "cpu")
                                  if f"{k[: -len('.weight')]}.weight_packed" in tensors else tensors[k].float()
                                  for k in twin.state_dict()}, strict=True)
            hooked = 0
            for name, module in twin.named_modules():
                declared = activations(checkpoint, name) if isinstance(module, torch.nn.Linear) else None
                if declared:
                    module.register_forward_pre_hook(lambda _, args, d=declared: (quantize_activations(args[0], *d),))
                    hooked += 1
            self.assertEqual(hooked, 14)   # 7 matrices in each of the 2 layers; the head is ignored
            with torch.no_grad():
                for (ids, first), (got, _, _), (plain, _, _) in zip(sequences(), quantized, exact):
                    logprobs = torch.log_softmax(twin(torch.tensor([ids])).logits[0].double(), -1)
                    self.assertEqual(got, [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))])
                    self.assertNotEqual(got, plain)


class patch_mode:
    """Score with StreamingReference in quality mode `mode` (test_eval_formats.scored opens it in preserve)."""

    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        from crit_eval.quality import reference
        self.original = reference.StreamingReference.__init__
        mode, original = self.mode, self.original

        def opened(self, model, device="cuda", quality_mode="preserve"):
            original(self, model, device, mode)
        reference.StreamingReference.__init__ = opened

    def __exit__(self, *exc):
        from crit_eval.quality import reference
        reference.StreamingReference.__init__ = self.original


if __name__ == "__main__":
    unittest.main()
