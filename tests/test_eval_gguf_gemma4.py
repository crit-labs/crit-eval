"""Gemma 4 GGUFs (architecture "gemma4"): the config.json their metadata derives and the reference's layout
(crit_eval.architectures.gguf_gemma4), the block formats a Gemma 4 GGUF mixes decoded as Core's own decoders decode
them, and the KV cache the limits count for keys equal to values.

The conformance test converts a tiny transformers Gemma 4 with GGUF's reference converter and needs a checkout of it
whose converter and gguf-py know gemma4 (CRIT_GGUF_TOOLS=/path/to/the/checkout) plus torch, transformers with
gemma4, safetensors, tokenizers and a gguf package that knows gemma4 (the checkout's gguf-py on PYTHONPATH will do);
without them it is skipped."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import GgufError, inspect_gguf_model  # noqa: E402
from crit_core.model.config import parse_model_config  # noqa: E402
from crit_core.model.gguf import TensorInfo  # noqa: E402
from crit_core.model.gguf_files import derive_config  # noqa: E402
from gguf_writer import ARRAY, BOOL, INT32, gguf_bytes  # noqa: E402
import test_eval_gguf as base  # noqa: E402   (by module: its own test classes stay in its own module)


def _has(*names):
    return all(importlib.util.find_spec(name) for name in names)


def _gguf_knows_gemma4():
    if not _has("gguf"):
        return False
    import gguf
    return "gemma4" in gguf.MODEL_ARCH_NAMES.values()


def _checkout_knows_gemma4(checkout):
    if not checkout:
        return False
    converter, constants = Path(checkout) / "conversion/gemma.py", Path(checkout) / "gguf-py/gguf/constants.py"
    return (converter.is_file() and "class Gemma4Model" in converter.read_text(encoding="utf-8")
            and constants.is_file() and '"gemma4"' in constants.read_text(encoding="utf-8"))


GGUF_TOOLS = os.environ.get("CRIT_GGUF_TOOLS")
HAS_GEMMA4 = _has("transformers") and importlib.util.find_spec("transformers.models.gemma4") is not None
HAS_DECODERS = base.HAS_LAYOUT and _gguf_knows_gemma4()
HAS_CONVERTER = base.HAS_CONVERTER and HAS_GEMMA4 and _gguf_knows_gemma4() and _checkout_knows_gemma4(GGUF_TOOLS)

LAYERS = 30
KINDS = ([True] * 5 + [False]) * 5   # sliding_window_pattern: five sliding layers, then a full one
# The metadata of a Gemma 4 MoE GGUF as the reference converter writes it (its vocabulary and chat template left
# out): sliding layers of 8 KV heads of 256, full layers of 2 KV heads of 512 whose keys are their values.
METADATA = [
    ("general.architecture", "gemma4"), ("gemma4.block_count", LAYERS), ("gemma4.context_length", 262144),
    ("gemma4.embedding_length", 2816), ("gemma4.feed_forward_length", 2112), ("gemma4.attention.head_count", 16),
    ("gemma4.attention.head_count_kv", [8 if s else 2 for s in KINDS], ARRAY, INT32),
    ("gemma4.rope.freq_base", 1e6), ("gemma4.rope.freq_base_swa", 1e4),
    ("gemma4.attention.layer_norm_rms_epsilon", 1e-6), ("gemma4.expert_count", 128),
    ("gemma4.expert_used_count", 8), ("gemma4.attention.key_length", 512), ("gemma4.attention.value_length", 512),
    ("gemma4.final_logit_softcapping", 30.0), ("gemma4.attention.sliding_window", 1024),
    ("gemma4.attention.shared_kv_layers", 0), ("gemma4.embedding_length_per_layer_input", 0),
    ("gemma4.attention.sliding_window_pattern", KINDS, ARRAY, BOOL), ("gemma4.attention.key_length_swa", 256),
    ("gemma4.attention.value_length_swa", 256), ("gemma4.expert_feed_forward_length", 704),
    ("gemma4.rope.dimension_count", 512), ("gemma4.rope.dimension_count_swa", 256),
    ("tokenizer.ggml.bos_token_id", 2), ("tokenizer.ggml.eos_token_id", 1), ("tokenizer.ggml.padding_token_id", 0)]
# The converter's rope_freqs for a full head of 512 at partial rotary factor 0.25: 64 rotated pairs, 192 not.
ROPE_FREQS = [1.0] * 64 + [1e30] * 192


def _tensors(full_values=False, routers=range(LAYERS)):
    out = [("token_embd.weight", (262144, 2816))]
    for i, sliding in enumerate(KINDS):
        out.append((f"blk.{i}.attn_k.weight", ((8 * 256) if sliding else (2 * 512), 2816)))
        if sliding or full_values:
            out.append((f"blk.{i}.attn_v.weight", ((8 * 256) if sliding else (2 * 512), 2816)))
    out += [(f"blk.{i}.ffn_gate_inp.weight", (128, 2816)) for i in routers]
    return out


def _changed(metadata, **values):
    keys = {"gemma4." + key.replace("__", "."): value for key, value in values.items()}
    out = [entry for entry in metadata if entry[0] not in keys]
    for key, value in keys.items():
        if value is not None:
            out.append((key, value, ARRAY, BOOL if isinstance(value[0], bool) else INT32)
                       if isinstance(value, list) else (key, value))
    return out


def gemma4_model(directory, metadata=METADATA, tensors=None, rope_freqs=ROPE_FREQS):
    """A GgufModel with `metadata`, a real rope_freqs tensor (its values are read) and the shapes of `tensors`."""
    path = Path(directory) / "gemma4.gguf"
    stored = [("rope_freqs.weight", "F32", (len(rope_freqs),), struct.pack(f"<{len(rope_freqs)}f", *rope_freqs))
              ] if rope_freqs else []
    path.write_bytes(gguf_bytes(metadata, stored))
    model = inspect_gguf_model(path)
    shapes = {name: TensorInfo(name, "Q8_0", shape, path, 0, 0) for name, shape in (tensors or _tensors())}
    return replace(model, tensors={**model.tensors, **shapes})


class Gemma4ConfigTests(unittest.TestCase):
    """The config rules over metadata alone (no converter)."""

    def test_the_metadata_derives_its_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = derive_config(gemma4_model(directory))
        stated = {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text", "vocab_size": 262144,
                  "hidden_size": 2816, "num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8,
                  "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 2, "attention_k_eq_v": True,
                  "num_kv_shared_layers": 0, "sliding_window": 1024, "max_position_embeddings": 262144,
                  "hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": 30.0,
                  "tie_word_embeddings": True, "intermediate_size": 2112, "use_double_wide_mlp": False,
                  "hidden_size_per_layer_input": 0, "enable_moe_block": True, "num_experts": 128,
                  "top_k_experts": 8, "moe_intermediate_size": 704, "bos_token_id": 2, "eos_token_id": 1,
                  "pad_token_id": 0,
                  "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                      "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                                         "rope_theta": 1000000.0}}}
        self.assertEqual({key: config.get(key) for key in stated}, stated)
        self.assertEqual(config["layer_types"], (["sliding_attention"] * 5 + ["full_attention"]) * 5)
        # Core's geometry reads the full layers' own heads and one cached tensor for their keys and values.
        model = parse_model_config(config)
        self.assertEqual(model.attention_shape(), (5, 16, 512, 2))
        self.assertEqual(model.kv_elements_per_token(), 5 * 2 * 512)
        self.assertEqual(model.moe.experts_per_token, 8)

    def test_full_layers_with_values_and_a_dense_model(self):
        metadata = _changed(METADATA, attention__head_count_kv=[8] * LAYERS, expert_count=None,
                            expert_used_count=None, expert_feed_forward_length=None)
        with tempfile.TemporaryDirectory() as directory:
            config = derive_config(gemma4_model(directory, metadata, _tensors(full_values=True, routers=())))
        self.assertEqual((config["attention_k_eq_v"], config["num_key_value_heads"], config["enable_moe_block"]),
                         (False, 8, False))
        self.assertNotIn("num_global_key_value_heads", config)
        self.assertNotIn("top_k_experts", config)

    def test_what_transformers_cannot_say_is_refused(self):
        cases = [
            (dict(attention__sliding_window_pattern=None), {}, "no per-layer gemma4.attention.sliding_window_pattern"),
            (dict(attention__sliding_window_pattern=[True] * LAYERS), {}, "the last layer is a sliding layer"),
            (dict(attention__value_length=256), {}, "value_length is 256, its key length 512"),
            (dict(attention__head_count_kv=[8, 4] * 15), {}, "holds \\[4, 8\\] KV head counts on the sliding"),
            ({}, dict(full_values=True), "2 KV heads and the sliding layers 8, but no shared keys"),
            ({}, dict(routers=range(10)), "some layers have a router"),
            ({}, dict(rope_freqs=[1.0] * 64 + [2.0] * 192), "rope_freqs.weight holds 256 factors"),
            ({}, dict(rope_freqs=None), "no rope_freqs.weight"),
        ]
        for values, options, message in cases:
            with self.subTest(message), tempfile.TemporaryDirectory() as directory:
                rope = options.pop("rope_freqs", ROPE_FREQS)
                with self.assertRaisesRegex(GgufError, message):
                    derive_config(gemma4_model(directory, _changed(METADATA, **values), _tensors(**options), rope))


def small_gemma4(path, architecture="gemma4"):
    """A small Gemma 4 GGUF (two sliding layers, then a full one whose keys are its values) at `path`, its metadata
    under `architecture`."""
    kinds, a = [True, True, False], architecture
    metadata = [("general.architecture", a), (f"{a}.block_count", 3), (f"{a}.context_length", 4096),
                (f"{a}.embedding_length", 64), (f"{a}.feed_forward_length", 128), (f"{a}.attention.head_count", 4),
                (f"{a}.attention.head_count_kv", [2, 2, 1], ARRAY, INT32), (f"{a}.attention.key_length", 32),
                (f"{a}.attention.value_length", 32), (f"{a}.attention.key_length_swa", 16),
                (f"{a}.attention.value_length_swa", 16), (f"{a}.attention.sliding_window", 8),
                (f"{a}.attention.sliding_window_pattern", kinds, ARRAY, BOOL),
                (f"{a}.attention.layer_norm_rms_epsilon", 1e-6)]
    tensors = [("rope_freqs.weight", "F32", (16,), struct.pack("<16f", *([1.0] * 4 + [1e30] * 12))),
               ("token_embd.weight", "F16", (32, 64)), ("output_norm.weight", "F32", (64,))]
    for i, sliding in enumerate(kinds):
        width, kv = (16, 2) if sliding else (32, 1)
        tensors += [(f"blk.{i}.attn_q.weight", "F16", (4 * width, 64)),
                    (f"blk.{i}.attn_k.weight", "F16", (kv * width, 64))]
        if sliding:
            tensors.append((f"blk.{i}.attn_v.weight", "F16", (kv * width, 64)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gguf_bytes(metadata, tensors))
    return path


class Gemma4LimitsTests(unittest.TestCase):
    """The ceiling and the batched step count the KV cache as Core does: a full layer whose keys are its values caches
    one tensor, and a sliding layer's window does not grow with the context."""

    def test_keys_equal_to_values_are_cached_once(self):
        from crit_eval.limits import batched, ceiling
        with tempfile.TemporaryDirectory() as directory:
            path = small_gemma4(Path(directory) / "gemma4-F16.gguf")
            self.assertTrue(ceiling.model_config(path).get("attention_k_eq_v"))
            # One full layer of one KV head of 32, keys and values one tensor: 32 elements, 64 bytes in BF16 (counted
            # as K and V it was 128).
            self.assertEqual(ceiling.model_ceiling(path)["kv_bytes_per_token"], 32 * 2)
            self.assertEqual(batched.batch_terms(path, 1)["kv_bytes_per_token"], 32)


class ScoringRefusalTests(unittest.TestCase):
    """What crit run create says before any candidate is built: whether the reference can score the model."""

    def test_a_model_the_reference_cannot_score_is_named_with_why(self):
        from unittest.mock import patch
        from crit_eval.architectures import gguf_layout
        from crit_eval.quality.checkpoints import scoring_refusal
        with tempfile.TemporaryDirectory() as directory:
            gemma4 = small_gemma4(Path(directory) / "gemma4" / "gemma4-F16.gguf")
            other = small_gemma4(Path(directory) / "other" / "gemma3-F16.gguf", architecture="gemma3")
            self.assertRegex(scoring_refusal(other), "'gemma3' has no config rule")   # Core cannot derive its config
            self.assertRegex(scoring_refusal(Path(directory) / "missing"), "missing")
            with patch.dict(gguf_layout.LAYOUTS, clear=True, llama=gguf_layout.LlamaLayout):
                self.assertRegex(scoring_refusal(gemma4), "'gemma4' has no layout rules .*supported: llama")
            refusal = scoring_refusal(gemma4)
            if HAS_DECODERS and HAS_GEMMA4:
                self.assertIsNone(refusal)
            else:   # what this machine lacks for it, by name
                self.assertRegex(refusal, "gguf|transformers")


# The block formats of a Gemma 4 GGUF's matrices; each block's FP16 scales sit where Core's decoders read them.
FORMATS = {"Q8_0": (34, 32, (0,)), "IQ4_NL": (18, 32, (0,)), "IQ4_XS": (136, 256, (0,)), "IQ3_S": (110, 256, (0,))}


@unittest.skipUnless(HAS_DECODERS, "needs torch, transformers, numpy and a gguf package that knows gemma4")
class Gemma4BlockFormatTests(unittest.TestCase):
    def test_each_format_decodes_as_cores_decoders_and_by_rows(self):
        import numpy as np
        import torch
        from crit_core.model.gguf_blocks import GgufTensors, TYPES, stored_values
        from crit_eval.architectures.gguf_layout import GgufCheckpoint
        generator = np.random.default_rng(4)
        rows, columns = 6, 512
        tensors = []
        for name, (block_bytes, values, scales) in FORMATS.items():
            data = generator.integers(0, 256, (rows, columns // values, block_bytes), dtype=np.uint8)
            for at in scales:   # finite scales of either sign
                data[..., at:at + 2] = (generator.uniform(-0.05, 0.05, data.shape[:2]).astype(np.float16)
                                        .view(np.uint8).reshape(*data.shape[:2], 2))
            tensors.append((f"blk.0.{name.lower()}.weight", name, (rows, columns), data.tobytes()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocks.gguf"
            path.write_bytes(gguf_bytes([("general.architecture", "gemma4"), ("gemma4.block_count", 1)], tensors))
            model = inspect_gguf_model(path)
            checkpoint = GgufCheckpoint(model, object())
            with GgufTensors(model) as reader:
                for name, dtype, _, _ in tensors:
                    with self.subTest(dtype):
                        cores = torch.from_numpy(stored_values(reader.rows(name), TYPES[dtype], columns))
                        whole = checkpoint.decode(name)
                        self.assertTrue(torch.equal(whole, cores))
                        self.assertTrue(torch.isfinite(whole).all())
                        index = torch.tensor([4, 0, 4, 5, 1])
                        self.assertTrue(torch.equal(checkpoint.rows(name, index), whole[index]))


# The tiny Gemma 4: three sliding layers and a full one whose keys are its values, with their own head dim and KV
# heads, and the MoE block (router, experts and the dense MLP beside them) on every layer.
TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True,
            layer_types=["sliding_attention"] * 3 + ["full_attention"], sliding_window=8,
            final_logit_softcapping=30.0, hidden_size_per_layer_input=0, enable_moe_block=True, num_experts=4,
            top_k_experts=2, moe_intermediate_size=32, max_position_embeddings=4096, bos_token_id=1, pad_token_id=0)
# The converter's Gemma 4 vocabulary is a SentencePiece-style one (LlamaHfVocab); the tiny model's byte-level BPE is
# written as GPT-2's, which leaves the tensors and the other metadata as they are.
CONVERT = """import runpy, sys
checkout = sys.argv[1]
sys.path[:0] = [checkout, checkout + "/gguf-py"]
import conversion.base as base
import conversion.gemma as gemma
base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"   # tiny test vocabularies have no known hash
gemma.Gemma4Model.set_vocab = lambda self: self._set_vocab_gpt2()
sys.argv = ["convert_hf_to_gguf.py"] + sys.argv[2:]
runpy.run_path(checkout + "/convert_hf_to_gguf.py", run_name="__main__")
"""


@unittest.skipUnless(HAS_CONVERTER, "needs CRIT_GGUF_TOOLS (a converter checkout whose converter and gguf-py know "
                                    "gemma4), torch, transformers with gemma4, safetensors, tokenizers and a gguf "
                                    "package that knows gemma4")
class Gemma4ConformanceTests(base.GgufConformanceTests):
    """The tiny Gemma 4, converted by the reference converter to an F32 GGUF, comes back tensor for tensor, the lone
    .gguf derives the original's config.json, and the reference scores it exactly as the model's own forward does."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        (cls.root / "convert-gemma4.py").write_text(CONVERT)

    def hf_model(self, name, model_type, options):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        out = self.root / f"hf-{name}"
        out.mkdir()
        self.tokenizer(out)
        # The converter cuts the embedding to the tokenizer's vocabulary (Gemma3Model.modify_tensors): the model has
        # exactly that many rows, so nothing is cut.
        tokenizer = json.loads((out / "tokenizer.json").read_text())
        vocab = len(tokenizer["model"]["vocab"]) + len(tokenizer["added_tokens"])
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, vocab_size=vocab, eos_token_id=0,
                                                                      **options), dtype=torch.float32)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.2 + (0.5 if "norm" in key or "scale" in key else 0))
            for layer in model.model.layers:   # a buffer the converter keeps (layer_output_scale)
                layer.layer_scalar.fill_(0.75)
        model.save_pretrained(out)
        return out

    def test_round_trips(self):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoModelForCausalLM
        from crit_core.model import open_model
        from crit_eval.architectures.gguf_layout import GgufCheckpoint
        from crit_eval.quality.checkpoints import _text_config
        from test_eval_formats import scored
        torch.manual_seed(1)
        sequences = [(torch.randint(0, 290, (40,)).tolist(), 5), (torch.randint(0, 290, (23,)).tolist(), 3)]
        hf = self.hf_model("gemma4", "gemma4_text", TINY)
        folder = self.root / "gguf-gemma4"
        folder.mkdir()
        out = folder / "gemma4-F32.gguf"
        subprocess.run([sys.executable, str(self.root / "convert-gemma4.py"), GGUF_TOOLS, str(hf), "--outfile",
                        str(out), "--outtype", "f32"], check=True, capture_output=True)
        weights = open_model(out).weights
        self.assertNotIn("blk.3.attn_v.weight", weights.tensors)   # the full layer's keys are its values
        self.assertIn("blk.0.ffn_gate_inp.scale", weights.tensors)
        self.assert_described_alike(out, hf)
        stored = load_file(hf / "model.safetensors")
        config = _text_config(out)
        full = config.per_layer_config[3]   # transformers' own reading of global_head_dim and its KV heads
        self.assertEqual((full.head_dim, full.num_key_value_heads), (32, 1))
        checkpoint = GgufCheckpoint(weights, config)
        with torch.device("meta"):
            skeleton = AutoModel.from_config(config)
        shapes = {"model." + k: v.shape for k, v in skeleton.state_dict().items()}
        self.assertIn("model.layers.0.router.per_expert_scale", shapes)
        for key, shape in shapes.items():
            expected = self.hf_tensor(stored, key).float()
            got = checkpoint.fit(key, checkpoint.raw(key), shape)
            self.assertTrue(torch.equal(got, expected), key)   # every move is exact: nothing rounds
        gguf_scores = scored(out, sequences)
        model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32, attn_implementation="eager",
                                                     experts_implementation="eager")
        with torch.no_grad():
            for (ids, first), (got, _, _) in zip(sequences, gguf_scores):
                logprobs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)
                self.assertEqual(got, [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))])


if __name__ == "__main__":
    unittest.main()
