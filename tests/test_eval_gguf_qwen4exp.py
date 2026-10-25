"""Qwen3.8-Flash-Next GGUFs (architecture "qwen4exp"): the config.json their metadata derives
(crit_core.model.gguf_qwen4exp) and the reference's layout (crit_eval.gguf_qwen4exp), with the n-gram table read by
rows (GgufCheckpoint.row_source).

The conformance test converts a tiny transformers Qwen4Exp with GGUF's reference converter and needs a checkout of it
whose converter and gguf-py know qwen4exp (CRIT_LLAMA_CPP=/path/to/the/checkout) plus torch, transformers with
qwen4_exp, safetensors, tokenizers and a gguf package that knows qwen4exp (the checkout's gguf-py on PYTHONPATH will
do); without them it is skipped."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import GgufError, inspect_gguf_model  # noqa: E402
from crit_core.model.gguf import TensorInfo  # noqa: E402
from crit_core.model.gguf_files import derive_config  # noqa: E402
from crit_core.model.gguf_qwen4exp import head_vocab_sizes, layer_multipliers  # noqa: E402
from gguf_writer import ARRAY, INT32, UINT64, gguf_bytes  # noqa: E402
import test_eval_gguf as base  # noqa: E402   (by module: its own test classes stay in its own module)


def _has(*names):
    return all(importlib.util.find_spec(name) for name in names)


def _transformers_has_qwen4exp():
    return _has("transformers") and importlib.util.find_spec("transformers.models.qwen4_exp") is not None


def _gguf_knows_qwen4exp():
    if not _has("gguf"):
        return False
    import gguf
    return "qwen4exp" in gguf.MODEL_ARCH_NAMES.values()


def _checkout_knows_qwen4exp(checkout):
    if not checkout:
        return False
    converter, constants = Path(checkout) / "conversion/qwen4exp.py", Path(checkout) / "gguf-py/gguf/constants.py"
    return converter.is_file() and constants.is_file() and '"qwen4exp"' in constants.read_text(encoding="utf-8")


LLAMA_CPP = os.environ.get("CRIT_LLAMA_CPP")
HAS_QWEN4EXP = _transformers_has_qwen4exp()
HAS_CONVERTER = (base.HAS_CONVERTER and HAS_QWEN4EXP and _gguf_knows_qwen4exp()
                 and _checkout_knows_qwen4exp(LLAMA_CPP))

# The PLE constants of the release (unsloth's Qwen3.8-Flash-Next GGUFs): seed 1234, ngram_vocab_size_base 20,000,000.
SIZES = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093,
         20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
# The metadata of a published Qwen3.8-Flash-Next GGUF (unsloth's UD-Q4_K_XL) the config rules read, with its value
# types (its vocabulary and chat template left out).
FLASH_NEXT_METADATA = [
    ("general.architecture", "qwen4exp"), ("qwen4exp.block_count", 48), ("qwen4exp.context_length", 262144),
    ("qwen4exp.embedding_length", 2560), ("qwen4exp.attention.head_count", 24),
    ("qwen4exp.attention.head_count_kv", 2), ("qwen4exp.rope.dimension_sections", [11, 11, 10, 0], ARRAY, INT32),
    ("qwen4exp.rope.freq_base", 1e7), ("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6),
    ("qwen4exp.expert_count", 512), ("qwen4exp.expert_used_count", 10), ("qwen4exp.attention.key_length", 256),
    ("qwen4exp.attention.value_length", 256), ("qwen4exp.expert_feed_forward_length", 640),
    ("qwen4exp.expert_shared_feed_forward_length", 640), ("qwen4exp.ssm.conv_kernel", 4),
    ("qwen4exp.ssm.state_size", 128), ("qwen4exp.ssm.group_count", 16), ("qwen4exp.ssm.time_step_rank", 48),
    ("qwen4exp.ssm.inner_size", 6144), ("qwen4exp.full_attention_interval", 4), ("qwen4exp.rope.dimension_count", 64),
    ("qwen4exp.hyper_connection.count", 4), ("qwen4exp.hyper_connection.low_rank", 320),
    ("qwen4exp.attention.indexer.head_count", 4), ("qwen4exp.attention.indexer.key_length", 128),
    ("qwen4exp.attention.indexer.top_k", 2048),
    ("qwen4exp.attention.compress_ratios", [0, 0, 0, 4] * 12, ARRAY, INT32),
    ("qwen4exp.ple.layers", [1], ARRAY, INT32), ("qwen4exp.ple.ngram_size", 3), ("qwen4exp.ple.heads_per_ngram", 8),
    ("qwen4exp.ple.conv_kernel", 4), ("qwen4exp.ple.eos_token_id", 248044),
    ("qwen4exp.embedding_length_per_layer_input", 160),
    ("qwen4exp.ple.layer_multipliers", [23703573157769, 20109073645365, 8052911324071], ARRAY, UINT64),
    ("qwen4exp.ple.head_offsets", [sum(SIZES[:h]) for h in range(16)], ARRAY, UINT64),
    ("qwen4exp.ple.head_vocab_sizes", SIZES, ARRAY, UINT64), ("qwen4exp.ple.image_token_id", 248056),
    ("tokenizer.ggml.eos_token_id", 248046), ("tokenizer.ggml.padding_token_id", 248044),
    ("tokenizer.ggml.bos_token_id", 248044)]
# What the release's tensor table says that the rules read: [name, row-major shape] (routers on every layer).
FLASH_NEXT_TENSORS = [("token_embd.weight", (248320, 2560)), ("output.weight", (248320, 2560)),
                      ("per_layer_token_embd.weight", (320001536, 160)), ("blk.1.ple_key.weight", (10240, 2560)),
                      *((f"blk.{i}.ffn_gate_inp.weight", (512, 2560)) for i in range(48)),
                      *((f"blk.{i}.indexer.k_proj.weight", (128, 2560)) for i in range(3, 48, 4))]


def _changed(metadata, **values):
    """`metadata` with the keys of `values` (qwen4exp.<key with dots as __>) replaced."""
    keys = {"qwen4exp." + key.replace("__", "."): value for key, value in values.items()}
    out = [entry for entry in metadata if entry[0] not in keys]
    for key, value in keys.items():
        out.append((key, value, ARRAY, UINT64 if "ple.head" in key or "multipliers" in key else INT32)
                   if isinstance(value, list) else (key, value))
    return out


class Qwen4ExpConfigTests(unittest.TestCase):
    """The config rules over metadata alone (no converter)."""

    def model(self, directory, metadata=FLASH_NEXT_METADATA, tensors=FLASH_NEXT_TENSORS):
        path = Path(directory) / "flash-next.gguf"
        path.write_bytes(gguf_bytes(metadata))
        return replace(inspect_gguf_model(path), tensors={name: TensorInfo(name, "Q8_0", shape, path, 0, 0)
                                                          for name, shape in tensors})

    def test_the_released_metadata_derives_its_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = derive_config(self.model(directory))
        stated = {"model_type": "qwen4_exp_text", "num_hidden_layers": 48, "hidden_size": 2560, "hc_count": 4,
                  "hc_lowrank": 320, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
                  "indexer_budget": 2048, "indexer_compress_ratio": 4, "ple_layer_ids": [2], "ple_embed_dim": 2560,
                  "ngram_size": 3, "heads_per_ngram": 8, "ple_conv_kernel_size": 4, "seed": 1234,
                  "ngram_vocab_size_base": 20000000, "make_ngram_vocab_size_divisible_by": 128,
                  "eos_token_id": 248044, "output_gate_type": "sigmoid", "norm_topk_prob": True,
                  "num_experts": 512, "shared_expert_intermediate_size": 640, "linear_num_value_heads": 48,
                  "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                      "mrope_section": [11, 11, 10], "mrope_interleaved": True}}
        self.assertEqual({key: config.get(key) for key in stated}, stated)
        self.assertEqual(config["layer_types"], (["linear_attention"] * 3 + ["full_attention"]) * 12)
        self.assertNotIn("attn_output_gate", config)
        if HAS_QWEN4EXP:   # transformers' own n-gram code computes the stored constants and table rows from it
            from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
            multipliers = modeling._build_layer_multipliers(config["vocab_size"], 3, 0, config["seed"]).tolist()
            self.assertEqual(multipliers, [23703573157769, 20109073645365, 8052911324071])
            sizes = [modeling._find_nth_prime_after(config["ngram_vocab_size_base"] - 1, h + 1) for h in range(16)]
            self.assertEqual(sizes, SIZES)
            self.assertEqual(-(-sum(sizes) // 128) * 128, 320001536)

    def test_hash_constants_of_other_settings_give_them_back(self):
        # A seed, n-gram base and padding other than the defaults: the seed is found by undoing the hash, the base
        # is the smallest that gives the same primes (any in (prime before the first, first] does), the divisor the
        # smallest that pads to the table's rows.
        vocab, seed, sizes = 248320, 987654321, list(head_vocab_sizes(1000, 16))
        metadata = _changed(FLASH_NEXT_METADATA, ple__layer_multipliers=list(layer_multipliers(vocab, 3, 0, seed)),
                            ple__head_vocab_sizes=sizes, ple__head_offsets=[sum(sizes[:h]) for h in range(16)])
        rows = -(-sum(sizes) // 1000) * 1000
        tensors = [(n, (rows, 160) if n == "per_layer_token_embd.weight" else s) for n, s in FLASH_NEXT_TENSORS]
        with tempfile.TemporaryDirectory() as directory:
            config = derive_config(self.model(directory, metadata, tensors))
        self.assertEqual(layer_multipliers(vocab, 3, 0, config["seed"]), layer_multipliers(vocab, 3, 0, seed))
        self.assertEqual(config["ngram_vocab_size_base"], 998)   # 997 is the prime before 1009
        self.assertEqual(head_vocab_sizes(config["ngram_vocab_size_base"], 16), tuple(sizes))
        divisor = config["make_ngram_vocab_size_divisible_by"]
        self.assertEqual(-(-sum(sizes) // divisor) * divisor, rows)

    def test_what_transformers_cannot_say_is_refused(self):
        cases = [
            # The converter's ratios when transformers renamed the layer types first: plain attention in the runtime.
            (dict(attention__compress_ratios=[0] * 48), None, "attention layers \\[3, 7, .*no compress ratio"),
            (dict(attention__compress_ratios=[0, 0, 0, 4, 0, 0, 0, 8] * 6), None, "holds \\[4, 8\\] QSA block sizes"),
            (dict(ple__layers=[1, 5]), None, "names 2 layers"),
            (dict(ple__layer_multipliers=[2, 4, 6]), None, "hash multipliers come from no seed"),
            (dict(ple__head_vocab_sizes=[20000003] * 16), None, "head offsets are not the running sums"),
            (dict(ple__head_vocab_sizes=[20000004] + SIZES[1:],
                  ple__head_offsets=[20000004 * (h > 0) + sum(SIZES[1:h]) for h in range(16)]), None,
             "head vocabularies are not consecutive primes"),
            ({}, ("per_layer_token_embd.weight", (100, 160)), "do not pad its 320001446 n-gram ids"),
            ({}, ("blk.3.indexer.k_proj.weight", (256, 2560)), "QSA has one key head of 128"),
        ]
        for values, tensor, message in cases:
            with self.subTest(message), tempfile.TemporaryDirectory() as directory:
                tensors = dict(FLASH_NEXT_TENSORS)
                if tensor:
                    tensors[tensor[0]] = tensor[1]
                with self.assertRaisesRegex(GgufError, message):
                    derive_config(self.model(directory, _changed(FLASH_NEXT_METADATA, **values), tensors.items()))


# The tiny model: three linear-attention layers (the second with PLE) and a QSA one, as released, plus an MTP block;
# a non-default seed (found again by undoing the hash) and n-gram base 48, the smallest that gives its head primes
# (53, 59, ...), so the derived config says what the original says; the sigmoid output gate, as the runtime applies.
TINY = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
            **base.EXPERTS, shared_expert_intermediate_size=48, hc_lowrank=8, ple_layer_ids=[2],
            ngram_vocab_size_base=48, seed=4321, split_ngram_parts=3, indexer_n_heads=2, indexer_kv_heads=1,
            indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4, output_gate_type="sigmoid",
            max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                           "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
                                                           "mrope_interleaved": True})
# The converter reads config.json through transformers when it can, whose qwen4_exp renames the released
# "full_attention" layer types to "indexed_attention", and then writes no compress ratio for them (refused above);
# read as written, as a converter without transformers' qwen4_exp does. "split" converts as converters before the
# shared-input merges (llama.cpp #31028) did, which wrote the published GGUFs: each half its own tensor.
CONVERT = """import json, runpy, sys
llama, form = sys.argv[1], sys.argv[2]
sys.path[:0] = [llama, llama + "/gguf-py"]
import conversion.base as base
base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"   # tiny test vocabularies have no known hash
base.ModelBase.load_hparams = staticmethod(lambda folder, *args, **kwargs:
                                           json.loads((folder / "config.json").read_text()))
if form == "split":
    import conversion.qwen4exp
    conversion.qwen4exp.Qwen4ExpTextModel._SHARED_INPUT_MERGES = ()
sys.argv = ["convert_hf_to_gguf.py"] + sys.argv[3:]
runpy.run_path(llama + "/convert_hf_to_gguf.py", run_name="__main__")
"""
MERGED = ("hc_attn_down_inject", "hc_ffn_down_inject", "attn_qkv_gate", "ssm_beta_alpha")
# The converter's elementwise transforms round once each way: norms 1 + w (PLE's too), A_log -exp(A_log).
ROUNDED = base.ROUNDED + ("norm_key.weight", "norm_query.weight", "norm_conv.weight")


@unittest.skipUnless(HAS_CONVERTER, "needs CRIT_LLAMA_CPP (a converter checkout whose converter and gguf-py know "
                                    "qwen4exp), torch, transformers with qwen4_exp, safetensors, tokenizers and a gguf "
                                    "package that knows qwen4exp")
class Qwen4ExpConformanceTests(base.GgufConformanceTests):
    """The tiny Qwen4Exp, converted by the reference converter to an F32 GGUF with and without its shared-input merges,
    comes back tensor for tensor (hash constants exactly), the lone .gguf derives the original's config.json and
    tokenizer, and the reference scores it as the model's own forward does, gathering both embedding tables by rows."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        (cls.root / "convert-qwen4exp.py").write_text(CONVERT)

    def hf_model(self, name, model_type, options):
        """The tiny model saved by transformers (its n-gram table in shards, experts one by one), with an MTP block
        and the released config's layer types."""
        import torch
        from safetensors.torch import load_file, save_file
        out = super().hf_model(name, model_type, options)
        tensors = load_file(out / "model.safetensors")
        for key, value in list(tensors.items()):
            if key.startswith("model.layers.3."):
                tensors["mtp.layers.0." + key[len("model.layers.3."):]] = value.clone()
        for part, shape in (("fc_embedding", (64, 64)), ("fc_hidden", (64, 64)), ("pre_fc_norm_embedding", (64,)),
                            ("pre_fc_norm_hidden", (64,)), ("hyper_connection_mixer.hc_norm", (256,)),
                            ("hyper_connection_mixer.input_mix_weight_down", (8, 256)),
                            ("hyper_connection_mixer.input_mix_weight_up", (256, 8))):
            tensors[f"mtp.{part}.weight"] = torch.randn(shape)
        save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
        config = json.loads((out / "config.json").read_text())
        config["layer_types"] = ["full_attention" if t == "indexed_attention" else t for t in config["layer_types"]]
        (out / "config.json").write_text(json.dumps({**config, "mtp_num_hidden_layers": 1}))
        return out

    def convert_form(self, form, hf):
        folder = self.root / f"gguf-qwen4exp-{form}"
        folder.mkdir()
        out = folder / f"qwen4exp-{form}-F32.gguf"
        subprocess.run([sys.executable, str(self.root / "convert-qwen4exp.py"), LLAMA_CPP, form, str(hf), "--outfile",
                        str(out), "--outtype", "f32"], check=True, capture_output=True)
        return out

    @staticmethod
    def hf_tensor(tensors, key):
        if key.endswith(".ngram_embedding.weight"):   # saved in shards, as released
            stem = key[: -len(".weight")]
            shards = sorted((int(k[len(stem) + 7:].split(".")[0]), k) for k in tensors
                            if k.startswith(stem + ".shard_"))
            import torch
            return torch.cat([tensors[k] for _, k in shards])
        return base.GgufConformanceTests.hf_tensor(tensors, key)

    def test_round_trips(self):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoModelForCausalLM
        from crit_core.model import open_model
        from crit_eval.gguf_layout import GgufCheckpoint
        from crit_eval.reference import _text_config
        from test_eval_formats import scored
        torch.manual_seed(1)
        sequences = [(torch.randint(0, 300, (40,)).tolist(), 5), (torch.randint(0, 300, (23,)).tolist(), 3)]
        sequences[0][0][17] = 0   # an end of sequence: the n-gram hash restarts after it
        hf = self.hf_model("qwen4exp", "qwen4_exp_text", TINY)
        stored = load_file(hf / "model.safetensors")
        # The model's own forward, with the experts' eager loop the reference runs (and transformers' own indexer
        # loop, which the reference's form selects exactly as, ties included).
        model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32, attn_implementation="eager",
                                                     experts_implementation="eager")

        def forward():
            with torch.no_grad():
                return [torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)[first - 1:-1]
                        .gather(1, torch.tensor(ids[first:])[:, None])[:, 0].tolist() for ids, first in sequences]
        original, weights_of_original = forward(), {k: v.clone() for k, v in model.state_dict().items()}
        for form in ("merged", "split"):
            with self.subTest(form):
                folder = self.convert_form(form, hf)
                weights = open_model(folder).weights
                merged = {part for name in weights.tensors for part in name.split(".")[2:3] if part in MERGED}
                self.assertEqual(merged, set(MERGED) if form == "merged" else set())
                self.assert_described_alike(folder, hf)
                config = _text_config(folder)
                checkpoint = GgufCheckpoint(weights, config)
                with torch.device("meta"):
                    skeleton = AutoModel.from_config(config)
                shapes = {"model." + k: v.shape for k, v in skeleton.state_dict().items()}
                shapes["lm_head.weight"] = stored["lm_head.weight"].shape
                for key, shape in shapes.items():
                    expected_tensor = self.hf_tensor(stored, key)
                    got = checkpoint.fit(key, checkpoint.raw(key), shape)
                    if expected_tensor.is_floating_point():
                        expected_tensor = expected_tensor.float()
                    if key.endswith(ROUNDED):
                        torch.testing.assert_close(got, expected_tensor, rtol=1e-6, atol=1e-7, msg=key)
                    else:   # moved data and the hash constants (int64 buffers from metadata) exactly
                        self.assertTrue(torch.equal(got, expected_tensor), key)
                # The reference gathers both embedding tables by rows and never decodes either whole.
                decoded, gathered = [], []
                decode, rows = GgufCheckpoint.decode, GgufCheckpoint.rows

                def logged_decode(self_, name):
                    decoded.append(name)
                    return decode(self_, name)

                def logged_rows(self_, name, index):
                    gathered.append(name)
                    return rows(self_, name, index)
                with patch.object(GgufCheckpoint, "decode", logged_decode), \
                        patch.object(GgufCheckpoint, "rows", logged_rows):
                    gguf_scores = scored(folder, sequences)
                tables = {"token_embd.weight", "per_layer_token_embd.weight"}
                self.assertEqual(tables & set(decoded), set())
                self.assertEqual(tables & set(gathered), tables)
                # Exactly the model's forward over the weights the GGUF holds; within the converter's one rounding
                # (A_log, whose exp the delta net takes, and the norms) of its forward over the original's.
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        value.copy_(checkpoint.fit(key, checkpoint.raw(key), value.shape))
                self.assertEqual([got for got, _, _ in gguf_scores], forward())
                for (got, _, _), want in zip(gguf_scores, original):
                    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
                model.load_state_dict(weights_of_original)


if __name__ == "__main__":
    unittest.main()
