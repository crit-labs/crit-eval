"""DeepSeek V4 GGUFs (architecture "deepseek4"): the config.json their metadata derives (crit_core.model.gguf_deepseek4)
and the reference's layout (crit_eval.gguf_deepseek4).

The conformance test writes a tiny checkpoint in DeepSeek's own format (FP8 matrices with E8M0 block scales, FP4
routed experts with E8M0 scales per 32 values, BF16 and F32 tensors, hash-routing tables), converts it with GGUF's
reference converter and needs a checkout of it whose converter and gguf-py know DeepSeek V4
(CRIT_LLAMA_CPP=/path/to/the/checkout) plus torch, transformers with deepseek_v4, safetensors, tokenizers and a gguf
package that knows deepseek4 (the checkout's gguf-py on PYTHONPATH will do); without them it is skipped. The same
checkpoint as safetensors in DeepSeek's own tensor names scores as transformers loads it (DeepseekV4NamesTests)."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import GgufError, inspect_gguf_model  # noqa: E402
from crit_core.model.gguf import TensorInfo  # noqa: E402
from crit_core.model.gguf_files import derive_config  # noqa: E402
from gguf_writer import ARRAY, FLOAT32, INT32, gguf_bytes  # noqa: E402


def _has(*names):
    return all(importlib.util.find_spec(name) for name in names)


def _transformers_has_deepseek4():
    if not _has("transformers"):
        return False
    from transformers import CONFIG_MAPPING
    return "deepseek_v4" in CONFIG_MAPPING


def _gguf_knows_deepseek4():
    if not _has("gguf"):
        return False
    import gguf
    return "deepseek4" in gguf.MODEL_ARCH_NAMES.values()


def _checkout_knows_deepseek4(checkout):
    if not checkout:
        return False
    converter, constants = Path(checkout) / "conversion/deepseek.py", Path(checkout) / "gguf-py/gguf/constants.py"
    return (converter.is_file() and "class DeepseekV4Model" in converter.read_text(encoding="utf-8")
            and constants.is_file() and '"deepseek4"' in constants.read_text(encoding="utf-8"))


LLAMA_CPP = os.environ.get("CRIT_LLAMA_CPP")
HAS_V4 = _transformers_has_deepseek4()
HAS_CONVERTER = (HAS_V4 and _has("torch", "safetensors", "tokenizers", "numpy") and _gguf_knows_deepseek4()
                 and _checkout_knows_deepseek4(LLAMA_CPP))

# The deepseek4.* metadata of a published DeepSeek-V4-Flash-0731 GGUF (unsloth's UD-Q4_K_XL), with its value types.
FLASH_METADATA = [
    ("general.architecture", "deepseek4"), ("deepseek4.block_count", 43), ("deepseek4.context_length", 1048576),
    ("deepseek4.embedding_length", 4096), ("deepseek4.attention.head_count", 64),
    ("deepseek4.attention.head_count_kv", 1), ("deepseek4.rope.scaling.type", "yarn"),
    ("deepseek4.rope.scaling.factor", 16.0), ("deepseek4.rope.scaling.original_context_length", 65536),
    ("deepseek4.rope.scaling.yarn_beta_fast", 32.0), ("deepseek4.rope.scaling.yarn_beta_slow", 1.0),
    ("deepseek4.rope.freq_base", 10000.0), ("deepseek4.attention.layer_norm_rms_epsilon", 1e-6),
    ("deepseek4.expert_count", 256), ("deepseek4.expert_used_count", 6), ("deepseek4.expert_gating_func", 4),
    ("deepseek4.attention.key_length", 512), ("deepseek4.attention.value_length", 512),
    ("deepseek4.rope.dimension_count", 64), ("deepseek4.attention.q_lora_rank", 1024),
    ("deepseek4.attention.sliding_window", 128), ("deepseek4.expert_feed_forward_length", 2048),
    ("deepseek4.expert_shared_count", 1), ("deepseek4.expert_weights_scale", 1.5),
    ("deepseek4.expert_weights_norm", True), ("deepseek4.swiglu_clamp_exp", [10.0] * 43, ARRAY, FLOAT32),
    ("deepseek4.swiglu_clamp_shexp", [10.0] * 43, ARRAY, FLOAT32), ("deepseek4.attention.indexer.head_count", 64),
    ("deepseek4.attention.indexer.key_length", 128), ("deepseek4.attention.indexer.top_k", 512),
    ("deepseek4.attention.output_group_count", 8), ("deepseek4.attention.output_lora_rank", 1024),
    ("deepseek4.attention.compress_ratios", [0, 0] + [4, 128] * 20 + [4, 0, 0, 0], ARRAY, INT32),
    ("deepseek4.attention.compress_rope_freq_base", 160000.0), ("deepseek4.hyper_connection.count", 4),
    ("deepseek4.hyper_connection.sinkhorn_iterations", 20), ("deepseek4.hyper_connection.epsilon", 1e-6),
    ("deepseek4.hash_layer_count", 3), ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "joyai-llm"),
    ("tokenizer.ggml.bos_token_id", 0), ("tokenizer.ggml.eos_token_id", 1), ("tokenizer.ggml.padding_token_id", 2),
]
# deepseek-ai/DeepSeek-V4-Flash's config.json, less what no GGUF metadata states: its storage (quantization_config,
# expert_dtype, torch_dtype), training (initializer_range) and the routing method transformers does not read.
FLASH_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"], "attention_bias": False, "bos_token_id": 0, "eos_token_id": 1,
    "hc_eps": 1e-06, "hc_mult": 4, "hc_sinkhorn_iters": 20, "head_dim": 512, "hidden_act": "silu", "hidden_size": 4096,
    "index_head_dim": 128, "index_n_heads": 64, "index_topk": 512, "max_position_embeddings": 1048576,
    "model_type": "deepseek_v4", "moe_intermediate_size": 2048, "n_routed_experts": 256, "n_shared_experts": 1,
    "norm_topk_prob": True, "num_attention_heads": 64, "num_experts_per_tok": 6, "num_hidden_layers": 43,
    "num_hash_layers": 3, "num_key_value_heads": 1, "num_nextn_predict_layers": 1, "o_groups": 8,
    "o_lora_rank": 1024, "q_lora_rank": 1024, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-06,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536,
                     "type": "yarn"},
    "rope_theta": 10000, "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus", "sliding_window": 128,
    "swiglu_limit": 10.0, "tie_word_embeddings": False, "use_cache": True, "vocab_size": 129280,
    "compress_rope_theta": 160000, "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
}
# Keys of a released config.json no GGUF metadata states; transformers' modeling code reads none of them.
UNSTATED = ("quantization_config", "expert_dtype", "torch_dtype", "dtype", "topk_method", "transformers_version")


def _stated(config):
    """A transformers DeepseekV4Config as a dict, less UNSTATED."""
    from transformers import CONFIG_MAPPING
    data = CONFIG_MAPPING["deepseek_v4"].from_dict({k: v for k, v in config.items() if k not in UNSTATED}).to_dict()
    return {k: v for k, v in data.items() if k not in UNSTATED}


class DeepseekV4ConfigTests(unittest.TestCase):
    """The config rules over metadata alone (no converter)."""

    def model(self, directory, metadata=FLASH_METADATA):
        path = Path(directory) / "flash.gguf"
        path.write_bytes(gguf_bytes(metadata))
        model = inspect_gguf_model(path)
        # The release's two tensor-table entries the rules read (its vocabulary rows and an untied head).
        tensors = {name: TensorInfo(name, "Q8_0", (129280, 4096), path, 0, 0)
                   for name in ("token_embd.weight", "output.weight")}
        return replace(model, tensors=tensors)

    def test_the_released_metadata_derives_the_released_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = derive_config(self.model(directory))
        for key, value in config.items():
            if key not in ("compress_ratios",):   # 0731's lists three entries past the 43 layers, Flash's one
                self.assertEqual(value, FLASH_CONFIG.get(key, value), key)
        self.assertEqual(config["compress_ratios"][:43], FLASH_CONFIG["compress_ratios"][:43])
        self.assertNotIn("num_nextn_predict_layers", config)   # the converter exports no MTP layer by default
        if HAS_V4:   # transformers builds the same model from either
            self.assertEqual(_stated(config), _stated(FLASH_CONFIG))

    def test_what_transformers_cannot_say_is_refused(self):
        cases = [("deepseek4.expert_weights_norm", False, "does not renormalise"),
                 ("deepseek4.attention.head_count_kv", 2, "shares one key/value head"),
                 ("deepseek4.expert_shared_count", 2, "exactly one shared expert"),
                 ("deepseek4.swiglu_clamp_shexp", [10.0] * 42 + [7.0], "one positive swiglu_limit"),
                 ("deepseek4.attention.compress_ratios", [0, 8] * 22, "does not give each of 43 layers"),
                 ("deepseek4.rope.scaling.type", "linear", "YaRN only"),
                 ("deepseek4.attention.sliding_window_pattern", 4, "metadata the config rules do not state")]
        for key, value, message in cases:
            with self.subTest(key), tempfile.TemporaryDirectory() as directory:
                metadata = [entry for entry in FLASH_METADATA if entry[0] != key]
                metadata.append((key, value, ARRAY, FLOAT32 if isinstance(value[0], float) else INT32)
                                if isinstance(value, list) else (key, value))
                with self.assertRaisesRegex(GgufError, message):
                    derive_config(self.model(directory, metadata))


# The tiny model: four layers (sliding, compressed sparse, heavily compressed, compressed sparse attention), two of
# them hash-routed, MTP's extra compress ratio as released; FP8 matrices up to two 128-row blocks.
TINY = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "vocab_size": 320,
        "hidden_size": 64, "moe_intermediate_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 64, "q_lora_rank": 32, "qk_rope_head_dim": 16, "n_routed_experts": 4,
        "num_experts_per_tok": 2, "n_shared_experts": 1, "scoring_func": "sqrtsoftplus", "norm_topk_prob": True,
        "routed_scaling_factor": 1.5, "max_position_embeddings": 4096, "rope_theta": 10000,
        "compress_rope_theta": 160000, "rope_scaling": {"type": "yarn", "factor": 16,
                                                        "original_max_position_embeddings": 256, "beta_fast": 32,
                                                        "beta_slow": 1},
        "compress_ratios": [0, 4, 128, 4, 0], "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "num_hash_layers": 2, "swiglu_limit": 2.0, "sliding_window": 8, "o_groups": 2, "o_lora_rank": 16,
        "index_n_heads": 2, "index_head_dim": 16, "index_topk": 4, "num_nextn_predict_layers": 1,
        "rms_norm_eps": 1e-6, "bos_token_id": 0, "eos_token_id": 1, "tie_word_embeddings": False,
        "attention_bias": False, "hidden_act": "silu", "initializer_range": 0.02, "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16", "expert_dtype": "fp4",
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}}
SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>", "<｜User｜>", "<｜Assistant｜>"]
# DeepSeek V3's (and V4's) pre-tokenizer splits, as its tokenizer.json has them.
SPLITS = [r"\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+",
          "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
          " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"]
CONVERT = """import runpy, sys
llama = sys.argv[1]
sys.path[:0] = [llama, llama + "/gguf-py"]
import conversion.base as base
base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "joyai-llm"   # DeepSeek V4's; tiny vocabularies hash anew
sys.argv = ["convert_hf_to_gguf.py"] + sys.argv[2:]
runpy.run_path(llama + "/convert_hf_to_gguf.py", run_name="__main__")
"""


def deepseek_checkpoint(out, fp8):
    """TINY in DeepSeek's own checkpoint format, its tensors small and random: dense matrices FP8 (E4M3 with E8M0
    scales per 128 x 128 block) when `fp8`, else BF16; the compressors' and indexer's projections, the router and
    the embeddings BF16 as released; hyper-connections, sinks, position biases and the routing bias F32; routed
    experts FP4 (E2M1 pairs in int8) with E8M0 scales per 32 values; tid2eid int32."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    c = TINY
    hidden, width, heads, head_dim, rank = c["hidden_size"], c["moe_intermediate_size"], c["num_attention_heads"], \
        c["head_dim"], c["q_lora_rank"]
    experts, streams, vocab = c["n_routed_experts"], c["hc_mult"], c["vocab_size"]
    generator = torch.Generator().manual_seed(0)
    tensors = {}

    def normal(shape, scale=0.2, offset=0.0):
        return torch.randn(shape, generator=generator) * scale + offset

    def e8m0(shape, low, high):
        return torch.randint(low, high, shape, generator=generator, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    def dense(name, rows, cols):
        if fp8:
            tensors[name + ".weight"] = normal((rows, cols), 3.0).to(torch.float8_e4m3fn)
            tensors[name + ".scale"] = e8m0((-(-rows // 128), -(-cols // 128)), 120, 125)
        else:
            tensors[name + ".weight"] = normal((rows, cols)).to(torch.bfloat16)

    def fp4(name, rows, cols):
        tensors[name + ".weight"] = torch.randint(-128, 128, (rows, cols // 2), generator=generator, dtype=torch.int8)
        tensors[name + ".scale"] = e8m0((rows, cols // 32), 118, 124)

    def bf16(name, *shape, offset=0.0):
        tensors[name] = normal(shape, offset=offset).to(torch.bfloat16)

    def f32(name, *shape, offset=0.0):
        tensors[name] = normal(shape, offset=offset)

    bf16("embed.weight", vocab, hidden)
    bf16("head.weight", vocab, hidden)
    bf16("norm.weight", hidden, offset=1.0)
    f32("hc_head_fn", streams, streams * hidden)
    f32("hc_head_base", streams)
    f32("hc_head_scale", 1, offset=1.0)
    for i in range(c["num_hidden_layers"]):
        layer, attn, ffn = f"layers.{i}.", f"layers.{i}.attn.", f"layers.{i}.ffn."
        bf16(layer + "attn_norm.weight", hidden, offset=1.0)
        bf16(layer + "ffn_norm.weight", hidden, offset=1.0)
        for part in ("hc_attn", "hc_ffn"):
            f32(f"{layer}{part}_fn", (2 + streams) * streams, streams * hidden)
            f32(f"{layer}{part}_base", (2 + streams) * streams)
            f32(f"{layer}{part}_scale", 3, offset=1.0)
        f32(attn + "attn_sink", heads)
        dense(attn + "wq_a", rank, hidden)
        bf16(attn + "q_norm.weight", rank, offset=1.0)
        dense(attn + "wq_b", heads * head_dim, rank)
        dense(attn + "wkv", head_dim, hidden)
        bf16(attn + "kv_norm.weight", head_dim, offset=1.0)
        dense(attn + "wo_a", c["o_groups"] * c["o_lora_rank"], heads * head_dim // c["o_groups"])
        dense(attn + "wo_b", hidden, c["o_groups"] * c["o_lora_rank"])
        ratio = c["compress_ratios"][i]
        if ratio:   # compressed sparse attention (4) compresses two overlapping series, heavily compressed one
            series = (2 if ratio == 4 else 1) * head_dim
            f32(attn + "compressor.ape", ratio, series)
            bf16(attn + "compressor.wkv.weight", series, hidden)
            bf16(attn + "compressor.wgate.weight", series, hidden)
            bf16(attn + "compressor.norm.weight", head_dim, offset=1.0)
        if ratio == 4:
            index_heads, index_dim = c["index_n_heads"], c["index_head_dim"]
            dense(attn + "indexer.wq_b", index_heads * index_dim, rank)
            bf16(attn + "indexer.weights_proj.weight", index_heads, hidden)
            f32(attn + "indexer.compressor.ape", ratio, 2 * index_dim)
            bf16(attn + "indexer.compressor.wkv.weight", 2 * index_dim, hidden)
            bf16(attn + "indexer.compressor.wgate.weight", 2 * index_dim, hidden)
            bf16(attn + "indexer.compressor.norm.weight", index_dim, offset=1.0)
        bf16(ffn + "gate.weight", experts, hidden)
        if i < c["num_hash_layers"]:
            tensors[ffn + "gate.tid2eid"] = torch.stack([torch.randperm(experts, generator=generator)
                                                         [:c["num_experts_per_tok"]] for _ in range(vocab)]).int()
        else:
            f32(ffn + "gate.bias", experts)
        for e in range(experts):
            fp4(f"{ffn}experts.{e}.w1", width, hidden)
            fp4(f"{ffn}experts.{e}.w2", hidden, width)
            fp4(f"{ffn}experts.{e}.w3", width, hidden)
        dense(ffn + "shared_experts.w1", width, hidden)
        dense(ffn + "shared_experts.w2", hidden, width)
        dense(ffn + "shared_experts.w3", width, hidden)
    out.mkdir(parents=True)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(TINY, indent=2))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(split), "isolated") for split in SPLITS]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=298, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(["the quick brown fox jumps over the lazy dog 12345 " * 4, "gguf models 東京 ひらがな"],
                                  trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": SPECIALS[0], "eos_token": SPECIALS[1],
        "pad_token": SPECIALS[2], "add_bos_token": False}))
    return out


def dequantized(original, out):
    """`original` as transformers loads it, in DeepSeek's names: each FP8 or FP4 matrix decoded with its scales by
    Fp8Dequantize, the first operation transformers' FP8 loader (dequantize=True) runs on each (weight, scale) pair
    before the model's own conversion mapping, which from_pretrained then applies. (The loader itself also needs
    accelerate.)"""
    import shutil
    import torch
    from safetensors.torch import load_file, save_file
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    stored = load_file(original / "model.safetensors")
    decode = Fp8Dequantize(None)
    plain = {}
    for key, value in stored.items():
        scale = key.removesuffix(".weight") + ".scale"
        if key.endswith(".scale"):
            continue
        plain[key] = (decode._dequantize_one(value, stored[scale], output_dtype=torch.float32) if scale in stored
                      else value)
    out.mkdir()
    save_file(plain, out / "model.safetensors", metadata={"format": "pt"})
    config = {k: v for k, v in TINY.items() if k not in ("quantization_config", "expert_dtype")}
    (out / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(original / name, out / name)
    return out


@unittest.skipUnless(HAS_V4 and _has("torch", "safetensors", "tokenizers"),
                     "needs torch, transformers with deepseek_v4, safetensors and tokenizers")
class DeepseekV4NamesTests(unittest.TestCase):
    """Safetensors in DeepSeek's own tensor names (hc_head_fn, layers.N.ffn.experts.E.w1, attn.attn_sink, ...), as
    released and as compressed-tensors quantizations of it keep them: the reference reads each parameter where
    transformers' renamings for deepseek_v4 find it and scores exactly as transformers scores the model it loads."""

    def test_deepseeks_own_names_score_as_transformers_loads_them(self):
        import torch
        from transformers import AutoModelForCausalLM
        from test_eval_formats import scored
        torch.manual_seed(1)
        sequences = [(torch.randint(0, 300, (150,)).tolist(), 5), (torch.randint(0, 300, (23,)).tolist(), 3)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = dequantized(deepseek_checkpoint(root / "original", False), root / "plain")
            model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, experts_implementation="eager")
            with torch.no_grad():
                for (ids, first), (got, _, _) in zip(sequences, scored(folder, sequences)):
                    logprobs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)
                    self.assertEqual(got, [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))])

    def test_a_second_pass_of_one_reference_scores_as_the_first(self):
        # The reference spread scores the long texts again on the same reference: every layer it released to meta
        # after the first pass, the compressors' own rotary frequencies too, loads again.
        import torch
        from crit_eval.reference import StreamingReference
        torch.manual_seed(1)
        ids = torch.randint(0, 300, (150,)).tolist()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = dequantized(deepseek_checkpoint(root / "original", False), root / "plain")
            scorer = StreamingReference(folder, device="cpu")
            first, again = (scorer.score_texts([ids], 5, None, root / f"scratch-{n}", log=lambda *_: None, long=[0])[0]
                            for n in range(2))
            for a, b in zip(first, again):
                self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(HAS_CONVERTER, "needs CRIT_LLAMA_CPP (a converter checkout whose converter and gguf-py know "
                                    "DeepSeek V4), torch, transformers with deepseek_v4, safetensors, tokenizers and "
                                    "a gguf that knows deepseek4")
class DeepseekV4ConformanceTests(unittest.TestCase):
    """The tiny checkpoint converted by the reference converter: the lone .gguf derives the original's config.json
    and tokenizer, every tensor comes back as transformers loads it from the original (the converter's Q8_0 matrices
    as gguf-py decodes what it stored, within a Q8_0 step of transformers' values; everything else bit for bit), and
    the reference scores it exactly as transformers scores those weights."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "convert.py").write_text(CONVERT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def convert(self, name, original):
        folder = self.root / f"gguf-{name}"
        folder.mkdir()
        out = folder / f"{name}.gguf"
        subprocess.run([sys.executable, str(self.root / "convert.py"), LLAMA_CPP, str(original), "--outfile", str(out)],
                       check=True, capture_output=True)
        return folder

    def assert_described_alike(self, folder, original):
        from tokenizers import Tokenizer
        from crit_core.model import open_model
        files = open_model(folder).files
        derived, stated = json.loads(files.read("config.json")), json.loads((original / "config.json").read_text())
        for key, value in derived.items():
            self.assertEqual(value, stated[key], key)
        self.assertEqual(_stated(derived), _stated(stated))
        tokenizers = [Tokenizer.from_str(files.read("tokenizer.json").decode()),
                      Tokenizer.from_file(str(original / "tokenizer.json"))]
        for text in ("the quick brown fox 1234567", "<｜User｜>gguf models 東京ひらがな<｜end▁of▁sentence｜>",
                     "  lazy\n\ndog's (fox)!? 12"):
            ids = [t.encode(text, add_special_tokens=False).ids for t in tokenizers]
            self.assertEqual(ids[0], ids[1], text)
        generation = json.loads(files.read("generation_config.json"))
        self.assertEqual([generation[f"{k}_token_id"] for k in ("bos", "eos", "pad")], [0, 1, 2])

    def test_round_trips(self):
        from collections import Counter
        import gguf
        import torch
        from transformers import AutoModelForCausalLM
        from crit_core.model import open_model
        from crit_eval.gguf_layout import GgufCheckpoint
        from crit_eval.reference import _text_config
        from test_eval_formats import scored
        torch.manual_seed(1)
        # 150 tokens cross the heavily compressed layer's 128-token window; the indexer keeps 4 of up to 37 entries.
        sequences = [(torch.randint(0, 300, (150,)).tolist(), 5), (torch.randint(0, 300, (23,)).tolist(), 3)]
        q8 = gguf.GGMLQuantizationType.Q8_0
        for name, fp8 in (("fp8", True), ("bf16", False)):
            with self.subTest(name):
                original = deepseek_checkpoint(self.root / f"original-{name}", fp8)
                folder = self.convert(name, original)
                self.assert_described_alike(folder, original)
                stored = open_model(folder).weights
                kinds = Counter(tensor.dtype for tensor in stored.tensors.values())
                self.assertEqual({k for k in kinds if kinds[k]}, {"MXFP4", "I32", "BF16", "F32"} | ({"Q8_0"} if fp8
                                                                                                     else set()))
                model = AutoModelForCausalLM.from_pretrained(dequantized(original, self.root / f"plain-{name}"),
                                                             dtype=torch.float32, experts_implementation="eager")
                state = model.state_dict()
                checkpoint = GgufCheckpoint(stored, _text_config(folder))
                quantized = 0
                for key, expected in state.items():
                    got = checkpoint.fit(key, checkpoint.raw(key), expected.shape)
                    if not expected.is_floating_point():   # tid2eid
                        self.assertTrue(torch.equal(got.long(), expected.long()), key)
                        continue
                    gguf_name = checkpoint.layout.gguf_name(key)
                    if gguf_name in stored.tensors and stored.tensors[gguf_name].dtype == "Q8_0":
                        # The converter quantized transformers' values with gguf-py; the stored block decodes as is.
                        blocks = gguf.quants.quantize(expected.numpy(), q8)
                        rounded = torch.from_numpy(gguf.quants.dequantize(blocks, q8).copy())
                        self.assertTrue(torch.equal(got, rounded), key)
                        step = expected.reshape(-1, 32).abs().amax(-1, keepdim=True) / 127
                        self.assertTrue(bool(((got - expected).reshape(-1, 32).abs() <= step).all()), key)
                        state[key] = got
                        quantized += 1
                    else:
                        self.assertTrue(torch.equal(got, expected), key)
                self.assertEqual(quantized > 0, fp8)
                model.load_state_dict(state)
                gguf_scores = scored(folder, sequences)
                with torch.no_grad():
                    for (ids, first), (got, _, _) in zip(sequences, gguf_scores):
                        logprobs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)
                        self.assertEqual(got, [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))])

    def test_long_texts_score_as_eager_attention(self):
        """A text marked long runs chunked_sink_attention (DeepSeek V4's sinks; its compressed entries after the
        sliding window, their block bias concatenated onto eager's additive mask) and scores exactly as transformers'
        eager forward: whole, in blocks of fewer query rows than the text, and resumed over PROMPT_CHUNK chunks as
        transformers resumes its own cache. (Chunks are compared with chunks: the indexer's top-k breaks ties between
        equal scores by position in the score row, so a text in chunks need not score as the text whole.)"""
        from unittest import mock
        import torch
        from transformers import AutoModelForCausalLM, DynamicCache
        from crit_core.model import open_model
        from crit_eval import reference
        from crit_eval.gguf_layout import GgufCheckpoint
        folder = self.convert("long", deepseek_checkpoint(self.root / "original-long", True))
        config = reference._text_config(folder)
        checkpoint = GgufCheckpoint(open_model(folder).weights, config)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
        model.load_state_dict({k: checkpoint.fit(k, checkpoint.raw(k), v.shape) for k, v in model.state_dict().items()})
        model.eval()
        torch.manual_seed(2)
        ids = torch.randint(0, 300, (300,)).tolist()   # past the 128-token compression and 8-token sliding windows

        def eager(chunk):
            cache, logits = DynamicCache(config=model.config), []
            with torch.no_grad():
                for start in range(0, len(ids), chunk):
                    piece = torch.tensor([ids[start:start + chunk]])
                    logits.append(model(piece, past_key_values=cache, use_cache=True).logits[0])
            logprobs = torch.log_softmax(torch.cat(logits).double(), -1)
            return [logprobs[i - 1, ids[i]].item() for i in range(1, len(ids))]

        def long(block, chunk):
            with mock.patch.multiple(reference, QUERY_BLOCK=block, PROMPT_CHUNK=chunk), \
                    tempfile.TemporaryDirectory() as scratch:
                scorer = reference.StreamingReference(folder, device="cpu")
                self.assertEqual(scorer.chunked, reference.CHUNKED_SINKS)
                return scorer.score_texts([ids], 5, None, scratch, log=lambda *_: None, long=[0])[0][0].tolist()

        whole = eager(len(ids))
        for block in (reference.QUERY_BLOCK, 32, 7):
            with self.subTest(block=block):
                self.assertEqual(long(block, reference.PROMPT_CHUNK), whole)
        self.assertEqual(long(16, 64), eager(64))


if __name__ == "__main__":
    unittest.main()
