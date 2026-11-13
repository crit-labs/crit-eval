"""Gemma 4 (GGUF architecture "gemma4"; transformers' gemma4_text) for crit-eval: the config.json its metadata derives
(gemma4_config, its rule in crit_eval.architectures.gguf_config) and the reference converter's layout undone for the
reference (Gemma4Layout, for crit_eval.architectures.gguf_layout).

config.json is the inverse of the reference converter's Gemma4Model.set_gguf_parameters (conversion/gemma.py, on top
of Gemma3Model's and TextModel's), for the model as GGUF's reference runtime defines it (src/models/gemma4.cpp):

    layer kinds          layer_types from attention.sliding_window_pattern (true: sliding_attention)
    head geometry        head_dim from attention.key_length_swa (the sliding layers'), global_head_dim from
                         attention.key_length (the full layers'); the value lengths must match their key lengths
    KV heads             attention.head_count_kv, one per layer: num_key_value_heads from the sliding layers,
                         num_global_key_value_heads from the full layers (each kind holds one value)
    keys equal values    attention_k_eq_v when no full layer that holds keys holds values (attn_v); the sliding
                         layers always hold both
    shared KV            num_kv_shared_layers from attention.shared_kv_layers
    PLE                  hidden_size_per_layer_input from embedding_length_per_layer_input (0: none),
                         vocab_size_per_layer_input from per_layer_token_embd's rows
    MLP                  intermediate_size from feed_forward_length; a per-layer list that doubles on the KV-shared
                         layers is use_double_wide_mlp
    experts              enable_moe_block when every layer has a router (ffn_gate_inp), with num_experts,
                         top_k_experts (expert_used_count) and moe_intermediate_size (expert_feed_forward_length)
    RoPE                 rope_parameters per layer kind: sliding "default" at rope.freq_base_swa (rotating
                         rope.dimension_count_swa of head_dim), full "proportional" at rope.freq_base, its
                         partial_rotary_factor from the rope_freqs tensor the converter generates for it (factor 1 on
                         the rotated pairs, 1e30 on the others)
    the rest             sliding_window, final_logit_softcapping, rms_norm_eps, the context length, the vocabulary
                         (token_embd's rows: the converter drops rows past the tokenizer's vocabulary), tied
                         embeddings (no output.weight) and the tokenizer's bos, eos and padding ids; hidden_activation
                         gelu_pytorch_tanh (the runtime's GELU is the tanh form)

Its tensors: Gemma 4's norms are stored as they are (the converter's norm shift is 0, unlike Gemma 3's 1 + w), the
embedding is cut to the tokenizer's vocabulary, layer_scalar is renamed layer_output_scale, and the router's two
vectors move: router.scale to ffn_gate_inp.scale, router.per_expert_scale to ffn_down_exps.scale. The fused experts
(gate_up_proj) map to ffn_gate_up_exps; a GGUF that keeps the halves (ffn_gate_exps, ffn_up_exps) reads gate then up.
The rope_freqs tensor is the converter's own and maps to no transformers tensor. Every move is undone exactly; nothing
rounds. The block formats (Q8_0, IQ4_NL, IQ4_XS, IQ3_S and the K-quants) decode through gguf-py like every layout's.
Proven by the round trip in tests/test_eval_gguf_gemma4.py.
"""
import re

from crit_core.model.gguf import GgufError

from .gguf_layout import Layout

# The factor the converter writes for a full layer's unrotated pairs (Gemma4Model.generate_extra_tensors).
UNROTATED = 1e30


def _kinds(model, get, layers):
    pattern = get(".attention.sliding_window_pattern")
    if not isinstance(pattern, tuple) or len(pattern) < layers:
        raise GgufError(f"the GGUF has no per-layer {model.architecture}.attention.sliding_window_pattern; Gemma 4's "
                        "layer kinds cannot be derived without it")
    return ["sliding_attention" if flag else "full_attention" for flag in pattern[:layers]]


def _per_kind(model, get, key, kinds, label):
    """{kind: the one value of `key` its layers share}: a per-layer list read by kind, a single value for both."""
    value = get(key)
    if value is None:
        return {}
    if not isinstance(value, tuple):
        return {kind: value for kind in set(kinds)}
    out = {}
    for kind in set(kinds):
        values = {v for v, k in zip(value, kinds) if k == kind}
        if len(values) != 1:
            raise GgufError(f"{model.architecture}{key} holds {sorted(values)} {label} on the {kind} layers; "
                            "transformers' Gemma 4 has one per layer kind")
        out[kind] = values.pop()
    return out


def _full_rotary_factor(model, head_dim):
    """partial_rotary_factor of the full layers' proportional RoPE, from the converter's rope_freqs tensor."""
    from .gguf_config import _tensor_f32
    tensor = model.tensors.get("rope_freqs.weight")
    if tensor is None:
        raise GgufError("the GGUF has no rope_freqs.weight; the converter writes it for Gemma 4's full layers, whose "
                        "partial rotary factor it states")
    factors = _tensor_f32(tensor)
    rotated = next((i for i, f in enumerate(factors) if f != 1.0), len(factors))
    if len(factors) != head_dim // 2 or any(f < UNROTATED * 0.99 for f in factors[rotated:]):
        raise GgufError(f"rope_freqs.weight holds {len(factors)} factors that are not 1 on rotated pairs then 1e30 on "
                        f"the others for a head of {head_dim}; Gemma 4's proportional RoPE writes that")
    return 2 * rotated / head_dim


def _keys_equal_values(model, kinds, shared):
    """attention_k_eq_v: whether the full layers that hold keys hold no values."""
    holding = [i for i, kind in enumerate(kinds) if kind == "full_attention" and i < len(kinds) - shared
               and f"blk.{i}.attn_k.weight" in model.tensors]
    values = {f"blk.{i}.attn_v.weight" in model.tensors for i in holding}
    if len(values) > 1:
        raise GgufError("some full-attention layers hold values (attn_v) and some do not; transformers' Gemma 4 "
                        "shares keys and values on every full layer or none")
    sliding = [i for i, kind in enumerate(kinds) if kind == "sliding_attention" and i < len(kinds) - shared
               and f"blk.{i}.attn_k.weight" in model.tensors and f"blk.{i}.attn_v.weight" not in model.tensors]
    if sliding:
        raise GgufError(f"sliding layers {sliding} hold keys but no values; Gemma 4 shares them on full layers only")
    return values == {False}


def _mlp(model, get, kinds, shared):
    """(intermediate_size, use_double_wide_mlp)."""
    widths = get(".feed_forward_length")
    if not isinstance(widths, tuple):
        return widths, False
    first_shared = len(kinds) - shared
    base = set(widths[:first_shared]) if first_shared else set(w // 2 for w in widths)
    if len(base) != 1:
        raise GgufError(f"{model.architecture}.feed_forward_length differs between layers ({sorted(set(widths))}); "
                        "Gemma 4 widens only its KV-shared layers' MLP")
    width = base.pop()
    later = set(widths[first_shared:len(kinds)])
    if later and later != {width} and later != {2 * width}:
        raise GgufError(f"the KV-shared layers' MLP widths {sorted(later)} are neither {width} nor {2 * width}")
    return width, bool(shared) and later == {2 * width}


def _experts(model, get, layers):
    routers = {f"blk.{i}.ffn_gate_inp.weight" in model.tensors for i in range(layers)}
    if routers == {False}:
        return {"enable_moe_block": False}
    if len(routers) > 1:
        raise GgufError("some layers have a router (ffn_gate_inp) and some do not; Gemma 4's MoE block is on every "
                        "layer or none")
    return {"enable_moe_block": True, "num_experts": get.require(".expert_count", "expert count"),
            "top_k_experts": get.require(".expert_used_count", "experts per token"),
            "moe_intermediate_size": get.require(".expert_feed_forward_length", "expert MLP width")}


def gemma4_config(model, get):
    """config.json for a gemma4 GgufModel (module docstring)."""
    if get(".nextn_predict_layers"):
        raise GgufError("the GGUF holds draft (nextn) blocks; Gemma 4's text model has none")
    layers = get.require(".block_count", "block count")
    heads = get.require(".attention.head_count", "attention head count")
    hidden = get.require(".embedding_length", "width")
    kinds = _kinds(model, get, layers)
    if kinds[-1] != "full_attention":
        raise GgufError("the last layer is a sliding layer; transformers' Gemma 4 makes it a full one")
    head_dim, full_dim = get.require(".attention.key_length_swa"), get.require(".attention.key_length")
    for key, dim in ((".attention.value_length_swa", head_dim), (".attention.value_length", full_dim)):
        if get.one(key) not in (None, dim):
            raise GgufError(f"{model.architecture}{key} is {get.one(key)}, its key length {dim}; Gemma 4's values "
                            "are as long as its keys")
    kv = _per_kind(model, get, ".attention.head_count_kv", kinds, "KV head counts")
    shared = get(".attention.shared_kv_layers") or 0
    equal = _keys_equal_values(model, kinds, shared)
    sliding_kv, full_kv = kv.get("sliding_attention", heads), kv.get("full_attention", heads)
    if not equal and full_kv != sliding_kv:
        raise GgufError(f"the full layers have {full_kv} KV heads and the sliding layers {sliding_kv}, but no shared "
                        "keys and values; transformers' Gemma 4 gives full layers their own KV heads only when they "
                        "share them")
    if any(f"blk.{i}.attn_q.bias" in model.tensors for i in range(layers)):
        raise GgufError("the GGUF has attention biases; Gemma 4 has none")
    width, double = _mlp(model, get, kinds, shared)
    rotary_swa = get(".rope.dimension_count_swa")
    sliding_rope = {"rope_type": "default", "rope_theta": get(".rope.freq_base_swa", 10000.0)}
    if rotary_swa and rotary_swa != head_dim:
        sliding_rope["partial_rotary_factor"] = rotary_swa / head_dim
    full_rope = {"rope_type": "proportional", "partial_rotary_factor": _full_rotary_factor(model, full_dim),
                 "rope_theta": get(".rope.freq_base", 1_000_000.0)}
    per_layer = get(".embedding_length_per_layer_input") or 0
    embedding = model.tensors.get("token_embd.weight")
    config = {
        "architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text",
        "vocab_size": embedding.shape[0] if embedding is not None else len(get("tokenizer.ggml.tokens") or ()) or None,
        "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": sliding_kv, "head_dim": head_dim, "global_head_dim": full_dim,
        "attention_k_eq_v": equal, "num_kv_shared_layers": shared, "layer_types": kinds,
        "sliding_window": get(".attention.sliding_window"), "max_position_embeddings": get(".context_length"),
        "rms_norm_eps": get(".attention.layer_norm_rms_epsilon"), "hidden_activation": "gelu_pytorch_tanh",
        "final_logit_softcapping": get(".final_logit_softcapping"), "attention_bias": False,
        "tie_word_embeddings": "output.weight" not in model.tensors,
        "rope_parameters": {"sliding_attention": sliding_rope, "full_attention": full_rope},
        "intermediate_size": width, "use_double_wide_mlp": double, "hidden_size_per_layer_input": per_layer,
        **_experts(model, get, layers)}
    if equal:
        config["num_global_key_value_heads"] = full_kv
    table = model.tensors.get("per_layer_token_embd.weight")
    if per_layer:
        if table is None:
            raise GgufError("the GGUF has per-layer inputs but no per_layer_token_embd.weight")
        config["vocab_size_per_layer_input"] = table.shape[0]
    for key, name in (("tokenizer.ggml.bos_token_id", "bos_token_id"), ("tokenizer.ggml.eos_token_id", "eos_token_id"),
                      ("tokenizer.ggml.padding_token_id", "pad_token_id")):
        if get(key) is not None:
            config[name] = get(key)
    return {key: value for key, value in config.items() if value is not None}


BLOCK_NAME = re.compile(r"model\.layers\.(\d+)\.(.+)$")
# HF tensor in a layer -> the GGUF tensor the converter moves it to (Gemma4Model.modify_tensors).
MOVED = {"router.scale": "ffn_gate_inp.scale", "router.per_expert_scale": "ffn_down_exps.scale"}


class Gemma4Layout(Layout):
    """Gemma4Model's tensors as transformers' Gemma 4 names and lays them out (module docstring)."""

    def compose(self, name, present):
        found = BLOCK_NAME.match(name)
        if found is None:
            return None
        layer, part = int(found[1]), found[2]
        if part in MOVED:
            moved = f"blk.{layer}.{MOVED[part]}"
            return ((moved,), lambda tensor: tensor) if moved in present else None
        if part == "experts.gate_up_proj":
            parts = (f"blk.{layer}.ffn_gate_exps.weight", f"blk.{layer}.ffn_up_exps.weight")
            if not all(p in present for p in parts):
                return None
            import torch
            return parts, lambda gate, up: torch.cat([gate, up], dim=1)
        return None
