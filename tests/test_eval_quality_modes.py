"""Quality modes: the checkpoint's own activation quantization (crit_eval.limits.schemes.activations, quantize_activations)
and the as_quantized reference that applies it (crit_eval.quality.reference), on hand-built values and a tiny NVFP4 model
written here (needs torch, transformers, safetensors and tokenizers)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        # long texts alone, is not judged by until the reference scores it again; a probed spread names every text.
        from crit_eval import load_json, EVAL_DIR
        from crit_eval.run import quality_checks
        protocol = load_json(EVAL_DIR / "protocol.json")
        self.assertEqual(protocol["quality"]["spread"]["from_tokens"], 0)
        prepared = {"quality": [{"id": "c", "set": "conversations", "prompt_ids": [1, 2], "continuation_ids": [3]},
                                {"id": "l", "set": "long-1024", "prompt_ids": [1] * 9000, "continuation_ids": [3]}]}
        identity = {"reference": 3}
        spread = {"conversations/prompt": {"kld_mean": 0.04, "kld_p99": 0.6, "ppl_ratio": 1.02}}
        with tempfile.TemporaryDirectory() as data:
            self.assertIsNone(quality_checks.stored_spread(data, prepared, protocol, identity))
            fixed.write_spread(data, {**identity, "spread_perturbation": "tf32", "spread_texts": ["l"]}, spread)
            self.assertIsNone(quality_checks.stored_spread(data, prepared, protocol, identity))   # the long text alone
            fixed.write_spread(data, {**identity, "spread_perturbation": "tf32", "spread_texts": ["c", "l"],
                                      "spread_skipped": ["l"]}, spread)
            self.assertEqual(quality_checks.stored_spread(data, prepared, protocol, identity), spread)

    def test_the_allowance_is_stored_beside_the_scores_under_their_identity(self):
        with tempfile.TemporaryDirectory() as data:
            fixed.write(data, {"reference": 3}, {"t": ([-0.5], [[1]], [[-0.5]])})
            allowance = {"activations": ["nvfp4"], "sets": {"long-1/prompt": {"kld_mean": 0.05}}}
            self.assertIsNone(fixed.load_allowance(data, {"reference": 3}))
            fixed.write_allowance(data, {"reference": 3, "allowance": "as_quantized"}, allowance)
            self.assertEqual(fixed.load_allowance(data, {"reference": 3}), allowance)
            self.assertIsNone(fixed.load_allowance(data, {"reference": 4}))   # another reference's
            self.assertTrue((Path(data) / fixed.STORE / fixed.ALLOWANCE).is_file())


@unittest.skipUnless(HAS_REFERENCE, "needs torch, transformers, safetensors and tokenizers")
class AsQuantizedReferenceTests(unittest.TestCase):
    def test_the_reference_quantizes_every_declared_input_as_the_checkpoint_does(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from crit_eval.limits.schemes import activations, quantize_activations
        from test_eval_formats import scored, sequences
        with tempfile.TemporaryDirectory() as temp:
            folder, tensors, config = nvfp4_checkpoint(Path(temp))
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

    def test_one_reference_run_stores_the_scores_and_the_as_quantized_allowance(self):
        # protocol quality.allowance: a checkpoint that declares quantized activations is scored a second time with
        # them quantized, and only how far that lands from the exact scores is kept, per set and path; scores an
        # earlier crit-eval stored for the as_quantized mode give the same allowance without the second run, and a
        # checkpoint with plain weights declares none.
        from crit_eval.quality import reference
        from crit_eval.reports import metrics
        from test_eval_formats import sequences
        with tempfile.TemporaryDirectory() as temp:
            folder, _, _ = nvfp4_checkpoint(Path(temp))
            texts = {f"t{i}": (ids, False) for i, (ids, _) in enumerate(sequences())}
            layout = {"t0": ("conversations", 5), "t1": ("long-23", 3)}
            measured = Path(temp) / "measured"
            exact = reference.StreamingReference(folder, "cpu")
            reference.store_texts(exact, texts, 4, None, measured, "m", log=lambda m: None)
            self.assertEqual(exact.declared_activations(), ["nvfp4"])
            identity = {**exact.identity("m"), **fixed.identity(texts, 4, None)}
            logged = []
            how = reference.store_allowance(folder, measured, texts, 4, None, identity, layout, ["nvfp4"],
                                            log=logged.append, device="cpu")
            self.assertEqual(how, "measured")
            allowance = fixed.load_allowance(measured, identity)
            self.assertEqual((allowance["activations"], allowance["source"]), (["nvfp4"], "measured"))
            self.assertEqual(sorted(allowance["sets"]), ["conversations/prompt", "conversations/served",
                                                         "long-23/prompt", "long-23/served"])
            # What it holds: the quantized run against the exact scores, as an engine's rows would be judged.
            quantized = reference.StreamingReference(folder, "cpu", quality_mode="as_quantized")
            rows = dict(zip(texts, quantized.score_texts([ids for ids, _ in texts.values()], 4, None,
                                                         Path(temp) / "scratch", log=lambda m: None)))
            stored = fixed.load(measured, identity)
            again = [{"logprob": lp, "top": [[int(i), float(v)] for i, v in zip(ids, tops)]}
                     for lp, ids, tops in zip(*(part.tolist() for part in rows["t0"]))]
            self.assertEqual(allowance["sets"]["conversations/served"],
                             metrics.text_quality(stored["t0"][4:], again[4:]))
            self.assertGreater(allowance["sets"]["conversations/served"]["kld_mean"], 0)
            self.assertTrue(any(line.startswith("reference allowance conversations/served") for line in logged))
            self.assertEqual(reference.store_allowance(folder, measured, texts, 4, None, identity, layout, ["nvfp4"],
                                                       log=lambda m: None, device="cpu"), "stored")
            # An earlier crit-eval's as_quantized store of the same texts: the same allowance, no second run.
            derived = Path(temp) / "derived"
            reference.store_texts(exact, texts, 4, None, derived, "m", log=lambda m: None)
            fixed.write(derived, {**identity, "quality_mode": "as_quantized"},
                        {name: tuple(part.tolist() for part in row) for name, row in rows.items()},
                        fixed.LEGACY_AS_QUANTIZED)
            with patch.object(reference.StreamingReference, "score_texts", side_effect=AssertionError("ran again")):
                self.assertEqual(reference.store_allowance(folder, derived, texts, 4, None, identity, layout,
                                                           ["nvfp4"], log=lambda m: None, device="cpu"), "derived")
            self.assertEqual(fixed.load_allowance(derived, identity)["sets"], allowance["sets"])
            # A declared form crit does not emulate: recorded, so an as_quantized run says why it has no allowance.
            refused = Path(temp) / "refused"
            reference.store_texts(exact, texts, 4, None, refused, "m", log=lambda m: None)
            from crit_eval.limits.schemes import SchemeError
            self.assertEqual(reference.store_allowance(folder, refused, texts, 4, None, identity, layout,
                                                       SchemeError("int4 activations"), log=lambda m: None),
                             "unsupported")
            self.assertEqual(fixed.load_allowance(refused, identity)["unsupported"], "int4 activations")
            # Plain weights declare nothing: an empty allowance, as_quantized judges as preserve.
            plain = Path(temp) / "original"
            self.assertEqual(reference.StreamingReference(plain, "cpu").declared_activations(), [])
            self.assertEqual(reference.store_allowance(plain, refused, texts, 4, None, {**identity, "model_sha256": "p"},
                                                       layout, [], log=lambda m: None), "none")
            self.assertEqual(fixed.load_allowance(refused, {"model_sha256": "p"}), {"activations": [], "sets": {}})

    def test_fused_experts_quantize_each_matrix_input_with_its_own_scale(self):
        # Experts the modeling code multiplies through fused parameters (gate and up joined, every expert stacked):
        # each expert's gate, up and down inputs are quantized with that matrix's own stored global scale.
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoConfig, AutoModelForCausalLM
        from crit_eval.limits.schemes import quantize_activations
        from crit_eval.quality import reference
        from test_eval_formats import scored, sequences, tokenizer
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / "original"
            original.mkdir()
            tokenizer(original)
            torch.manual_seed(0)
            experts = 4
            config = AutoConfig.for_model("qwen3_moe", vocab_size=320, eos_token_id=0, hidden_size=64,
                                          intermediate_size=128, moe_intermediate_size=32, num_experts=experts,
                                          num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=4,
                                          num_key_value_heads=2, head_dim=16, tie_word_embeddings=False)
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
            with torch.no_grad():
                for key, parameter in model.named_parameters():
                    parameter.copy_(torch.randn_like(parameter) * 0.05 + (1.0 if "norm" in key else 0))
            model.save_pretrained(original)
            stored = load_file(str(original / "model.safetensors"))
            folder = Path(temp) / "nvfp4"
            folder.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (folder / name).write_bytes((original / name).read_bytes())
            tensors, scales = {}, {}
            for key, value in stored.items():
                base = key[: -len(".weight")]
                if ".mlp.experts." in key:
                    tensors.update({f"{base}.{k}": v for k, v in nvfp4_matrix(value, 448.0 * 6 / 0.2).items()})
                    # A scale of its own per expert and matrix, so a wrong pairing changes the scores.
                    scales[base] = 448.0 * 6 / (1.0 + len(scales) % 7)
                    tensors[f"{base}.input_global_scale"] = torch.tensor([scales[base]])
                else:
                    tensors[key] = value.contiguous()
            self.assertEqual(len(scales), 2 * experts * 3)
            save_file(tensors, str(folder / "model.safetensors"))
            saved = json.loads((original / "config.json").read_text())
            saved["quantization_config"] = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                            "config_groups": {"group_0": {"targets": ["re:.*mlp\\.experts.*"],
                                                                          "weights": {"num_bits": 4, "type": "float",
                                                                                      "group_size": 16},
                                                                          "input_activations": NVFP4_INPUTS}},
                                            "ignore": ["lm_head"]}
            (folder / "config.json").write_text(json.dumps(saved))
            with patch_mode("as_quantized"), patch.object(reference.StreamingReference, "stream_experts", True):
                streamed = scored(folder, sequences())
            with patch_mode("as_quantized"), patch.object(reference.StreamingReference, "stream_experts", False):
                whole = scored(folder, sequences())
            exact = scored(folder, sequences())
            # The twin: the decoded weights in transformers' eager experts loop, every matrix's input quantized alike.
            from crit_eval.quality.checkpoints import _text_config, open_checkpoint
            checkpoint = open_checkpoint(folder, _text_config(folder))
            twin = AutoModelForCausalLM.from_config(config, dtype=torch.float32, attn_implementation="eager")
            state = {}
            for key in twin.state_dict():
                if key.endswith("experts.gate_up_proj") or key.endswith("experts.down_proj"):
                    layer = key[: -len("gate_up_proj" if key.endswith("gate_up_proj") else "down_proj")]
                    parts = ("gate_proj", "up_proj") if key.endswith("gate_up_proj") else ("down_proj",)
                    state[key] = torch.stack([torch.cat([checkpoint.dense(f"{layer}{e}.{part}", "cpu") for part in parts])
                                              for e in range(experts)])
                else:
                    state[key] = stored[key].float()
            twin.load_state_dict(state, strict=True)

            def quantized_loop(module, layer):
                def forward(hidden_states, top_k_index, top_k_weights):
                    out = torch.zeros_like(hidden_states)
                    mask = torch.nn.functional.one_hot(top_k_index, num_classes=experts + 1).permute(2, 1, 0)
                    for (e,) in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().tolist():
                        if e == experts:
                            continue
                        position, token = torch.where(mask[e])
                        x = hidden_states[token]
                        q = lambda v, part: quantize_activations(v, "nvfp4", scales[f"{layer}{e}.{part}"])
                        rows = module.gate_up_proj[e]
                        gate = torch.nn.functional.linear(q(x, "gate_proj"), rows[:32])
                        up = torch.nn.functional.linear(q(x, "up_proj"), rows[32:])
                        h = module.act_fn(gate) * up
                        h = torch.nn.functional.linear(q(h, "down_proj"), module.down_proj[e])
                        out.index_add_(0, token, h * top_k_weights[token, position, None])
                    return out
                return forward
            for name, module in twin.named_modules():
                if name.endswith("mlp.experts"):
                    module.forward = quantized_loop(module, name + ".")
            with torch.no_grad():
                for (ids, first), got, again, (plain, _, _) in zip(sequences(), streamed, whole, exact):
                    logprobs = torch.log_softmax(twin(torch.tensor([ids])).logits[0].double(), -1)
                    self.assertEqual(got[0], [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))])
                    self.assertEqual(got, again)
                    self.assertNotEqual(got[0], plain)


def nvfp4_checkpoint(root):
    """test_eval_formats.tiny's model stored as compressed-tensors NVFP4 with NVFP4 input activations on every decoder
    matrix (the head left plain), in root / "nvfp4": (folder, its tensors, its config)."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from test_eval_formats import tiny
    original = tiny(root / "original")
    model = AutoModelForCausalLM.from_pretrained(original, dtype=torch.float32)
    folder = root / "nvfp4"
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
    return folder, tensors, config


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
