"""config.json for the GGUF architectures the reference reads, from their metadata alone (standard library only).

CONFIG_RULES gives, per general.architecture, the inverse of GGUF's reference converter's set_gguf_parameters for
it, for the model as the converter's reference runtime defines it (where that runtime fixes a behaviour for an
architecture, such as Qwen3 MoE always renormalising its top-k expert weights, the config states it). crit-core
derives a lone GGUF's config.json with them (crit_core.model.gguf_files, through crit_core.registry gguf_config:
crit_eval registers CONFIG_RULES when imported and names it in an entry point), and crit_eval.architectures.gguf_layout
inverts the converter's tensor transforms for the same architectures. A rule builds on crit-core's generic keys
(base_config, expert_config, layer_flag); DeepSeek V4's, Qwen3.8-Flash-Next's and Gemma 4's own keys are
crit_eval.architectures.gguf_deepseek4, gguf_qwen4exp and gguf_gemma4. Every rule is tested against the original
config.json of models converted by the reference converter (tests/test_eval_gguf.py, test_eval_gguf_deepseek4.py,
test_eval_gguf_qwen4exp.py and test_eval_gguf_gemma4.py).
"""
import math
import struct

from crit_core.model.gguf import GgufError
from crit_core.model.gguf_files import base_config, decimal, expert_config, layer_flag


def _tensor_f32(tensor):
    """The values of a small F32 tensor (rope_freqs), read from its file."""
    if tensor.dtype != "F32" or tensor.nbytes > 1 << 20:
        raise GgufError(f"{tensor.name}: expected a small F32 tensor, found {tensor.dtype} of {tensor.nbytes} bytes")
    with open(tensor.shard, "rb") as handle:
        handle.seek(tensor.data_begin)
        data = handle.read(tensor.nbytes)
    if len(data) != tensor.nbytes:
        raise GgufError(f"{tensor.name}: the file ends inside the tensor")
    return struct.unpack(f"<{tensor.nbytes // 4}f", data)


def _llama3_rope(factors, theta, head_dim):
    """The rope_parameters of Llama 3's frequency scaling whose per-dimension factors (the GGUF's rope_freqs tensor,
    written by LlamaModel.generate_extra_tensors) are `factors`: the factor is their largest value; the original
    context and the low and high frequency factors are the released values that reproduce every factor."""
    factor = max(factors)
    freqs = [1.0 / (theta ** (i / head_dim)) for i in range(0, head_dim, 2)]
    for original in (8192, 4096, 2048, 16384, 32768, 65536, 131072):
        for low in (1.0, 2.0, 0.5):
            for high in (4.0, 8.0, 2.0, 1.0):
                if high <= low:
                    continue
                expected = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < original / high:
                        expected.append(1.0)
                    elif wavelen > original / low:
                        expected.append(factor)
                    else:
                        smooth = (original / wavelen - low) / (high - low)
                        expected.append(1 / ((1 - smooth) / factor + smooth))
                if len(expected) == len(factors) and all(math.isclose(a, b, rel_tol=1e-5)
                                                         for a, b in zip(expected, factors)):
                    return {"rope_type": "llama3", "factor": decimal(factor), "low_freq_factor": low,
                            "high_freq_factor": high, "original_max_position_embeddings": original}
    raise GgufError("rope_freqs.weight holds frequency factors no Llama 3 rope scaling reproduces; the config rules "
                    "cannot state them")


def _rope(get, model, head_dim, rotary_dim=None):
    """rope_parameters from <arch>.rope.*: the base, and linear, YaRN or (with a rope_freqs tensor) Llama 3 scaling."""
    rope = {"rope_type": "default", "rope_theta": get(".rope.freq_base", 10000.0)}
    kind = get(".rope.scaling.type")
    if kind in ("linear", "yarn"):
        rope.update(rope_type=kind, factor=get(".rope.scaling.factor"))
        if kind == "yarn":
            rope["original_max_position_embeddings"] = get(".rope.scaling.original_context_length")
            for key, name in ((".rope.scaling.yarn_beta_fast", "beta_fast"), (".rope.scaling.yarn_beta_slow",
                              "beta_slow"), (".rope.scaling.yarn_attn_factor", "attention_factor"),
                              (".rope.scaling.yarn_ext_factor", "extrapolation_factor"),
                              (".rope.scaling.yarn_truncate", "truncate")):
                if get(key) is not None:
                    rope[name] = get(key)
    elif kind not in (None, "none"):
        raise GgufError(f"{model.architecture}.rope.scaling.type {kind!r} has no config rule")
    freqs = model.tensors.get("rope_freqs.weight")
    if freqs is not None:
        if kind not in (None, "none"):
            raise GgufError("the GGUF has rope_freqs.weight and rope.scaling.type; the config rules take one of them")
        rope.update(_llama3_rope(_tensor_f32(freqs), rope["rope_theta"], rotary_dim or head_dim))
    return rope


def _llama(model, get):
    config, layers, _ = base_config(model, get)
    rotary = get(".rope.dimension_count")
    if rotary is not None and rotary != config["head_dim"]:
        raise GgufError(f"llama rotates {rotary} of {config['head_dim']} head dimensions; Llama rotates all")
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", **config,
            "rope_parameters": _rope(get, model, config["head_dim"]),
            "attention_bias": layer_flag(model, "attn_q.bias", layers, "attention bias"),
            "mlp_bias": layer_flag(model, "ffn_up.bias", layers, "MLP bias")}


def _qwen2(model, get):
    config, layers, _ = base_config(model, get)
    if not layer_flag(model, "attn_q.bias", layers, "attention bias"):
        raise GgufError("a Qwen2 GGUF without query/key/value biases is not Qwen2 (its projections always have them)")
    return {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", **config,
            "rope_parameters": _rope(get, model, config["head_dim"]), "use_sliding_window": False}


def _qwen3(model, get):
    config, layers, _ = base_config(model, get)
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", **config,
            "rope_parameters": _rope(get, model, config["head_dim"]),
            "attention_bias": layer_flag(model, "attn_q.bias", layers, "attention bias"), "use_sliding_window": False}


def _qwen3moe(model, get):
    config, layers, _ = base_config(model, get)
    experts, dense = expert_config(model, get, layers)
    # The reference runtime's qwen3moe graph always renormalises the top-k weights (build_moe_ffn norm_w).
    return {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", **config, **experts,
            "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": dense,
            "rope_parameters": _rope(get, model, config["head_dim"]),
            "attention_bias": layer_flag(model, "attn_q.bias", layers, "attention bias"), "use_sliding_window": False}


def _qwen35(model, get, moe=False):
    config, layers, drafts = base_config(model, get)
    count = config["num_hidden_layers"]
    recurrent = get(".attention.recurrent_layers")
    if recurrent is not None:
        kinds = ["linear_attention" if flag else "full_attention" for flag in recurrent[:count]]
    else:
        interval = get(".full_attention_interval", 4)
        kinds = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    state, value_heads = get.require(".ssm.state_size"), get.require(".ssm.time_step_rank")
    inner = get.require(".ssm.inner_size")
    rotary = get(".rope.dimension_count")
    rope = _rope(get, model, config["head_dim"])
    rope["partial_rotary_factor"] = rotary / config["head_dim"] if rotary else 0.25
    sections = get(".rope.dimension_sections")
    if sections:
        rope["mrope_section"] = list(sections[:3])
        rope["mrope_interleaved"] = True
    out = {"architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
           "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text", **config,
           "layer_types": kinds, "full_attention_interval": get(".full_attention_interval", 4),
           "linear_conv_kernel_dim": get.require(".ssm.conv_kernel"), "linear_key_head_dim": state,
           "linear_value_head_dim": inner // value_heads, "linear_num_key_heads": get.require(".ssm.group_count"),
           "linear_num_value_heads": value_heads, "rope_parameters": rope, "attn_output_gate": True,
           "attention_bias": False, "mtp_num_hidden_layers": drafts}
    if moe:
        experts, dense = expert_config(model, get, layers)
        if dense:
            raise GgufError(f"Qwen3.5 MoE layers {dense} have no router; every layer of the architecture has experts")
        out.update(experts)
        shared = get(".expert_shared_feed_forward_length")
        if shared is not None:
            out["shared_expert_intermediate_size"] = shared
        out.pop("intermediate_size", None)
    return out


def _deepseek4(model, get):
    from .gguf_deepseek4 import deepseek4_config
    return deepseek4_config(model, get)


def _gemma4(model, get):
    from .gguf_gemma4 import gemma4_config
    return gemma4_config(model, get)


def _qwen4exp(model, get):
    from .gguf_qwen4exp import qwen4exp_config
    return qwen4exp_config(model, get, _qwen35(model, get, moe=True))


# general.architecture -> the inverse of the converter's set_gguf_parameters for it (gguf_layout.LAYOUTS holds the
# inverse of its tensor transforms for the same architectures).
CONFIG_RULES = {
    "llama": _llama,
    "qwen2": _qwen2,
    "qwen3": _qwen3,
    "qwen3moe": _qwen3moe,
    "qwen35": _qwen35,
    "qwen35moe": lambda model, get: _qwen35(model, get, moe=True),
    "deepseek4": _deepseek4,
    "qwen4exp": _qwen4exp,
    "gemma4": _gemma4,
}
