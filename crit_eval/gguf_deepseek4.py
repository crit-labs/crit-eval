"""DeepSeek V4 (GGUF architecture "deepseek4") read as transformers' DeepseekV4 checkpoint, for Eval's reference.

GGUF's reference converter (DeepseekV4Model in conversion/deepseek.py) reads DeepSeek's own checkpoint naming
(layers.N.attn.wq_a, ffn.experts.E.w1, hc_attn_fn, ...) and names each tensor with its own table, not TensorNameMap;
transformers' conversion mapping for deepseek_v4 renames the same checkpoint to its modeling code's names. TENSORS
and LAYER_TENSORS are the two composed: a modeling-code name and the GGUF tensor the converter writes for the same
DeepSeek tensor. The converter treats the tensors four ways:

    renamed     every tensor but the routed experts and the hash-routing tables: 1-D BF16 tensors widened to F32
                (exactly), BF16 and F32 matrices kept, FP8 matrices dequantized with their E8M0 block scales and
                stored Q8_0, which gguf-py decodes as stored
    stacked     each layer's routed experts' FP4 w1, w2, w3 with their E8M0 scales repacked bit for bit to MXFP4
                ffn_{gate,down,up}_exps [experts, rows, columns]; transformers' gate_up_proj is gate then up
    integers    the hash-routed layers' tid2eid tables (token -> experts) as I32 ffn_gate_tid2eid, a persistent
                buffer of the router returned as integers
    dropped     the hash-routed layers' unused gate.bias, the vision-token bias exp_probs_b_vl and the MTP layers,
                which the text model never reads

The round trip in tests/test_eval_gguf_deepseek4.py proves the table: a tiny checkpoint in DeepSeek's format
converted by the reference converter comes back tensor for tensor and scores alike.
"""
import re

from .gguf_layout import ExpertLayout, GgufLayoutError

# Modeling-code name (under "model.", per layer under layers.N.) -> GGUF tensor (per layer under blk.N.).
TENSORS = {
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "hc_head.hc_fn": "output_hc_fn.weight",
    "hc_head.hc_base": "output_hc_base.weight",
    "hc_head.hc_scale": "output_hc_scale.weight",
}
LAYER_TENSORS = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "attn_hc.fn": "hc_attn_fn.weight",
    "attn_hc.base": "hc_attn_base.weight",
    "attn_hc.scale": "hc_attn_scale.weight",
    "ffn_hc.fn": "hc_ffn_fn.weight",
    "ffn_hc.base": "hc_ffn_base.weight",
    "ffn_hc.scale": "hc_ffn_scale.weight",
    "self_attn.sinks": "attn_sinks.weight",
    "self_attn.q_a_proj.weight": "attn_q_a.weight",
    "self_attn.q_a_norm.weight": "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight",
    "self_attn.kv_proj.weight": "attn_kv.weight",
    "self_attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "self_attn.o_a_proj.weight": "attn_output_a.weight",
    "self_attn.o_b_proj.weight": "attn_output_b.weight",
    "self_attn.compressor.position_bias": "attn_compressor_ape.weight",
    "self_attn.compressor.kv_proj.weight": "attn_compressor_kv.weight",
    "self_attn.compressor.gate_proj.weight": "attn_compressor_gate.weight",
    "self_attn.compressor.kv_norm.weight": "attn_compressor_norm.weight",
    "self_attn.compressor.indexer.position_bias": "indexer_compressor_ape.weight",
    "self_attn.compressor.indexer.kv_proj.weight": "indexer_compressor_kv.weight",
    "self_attn.compressor.indexer.gate_proj.weight": "indexer_compressor_gate.weight",
    "self_attn.compressor.indexer.kv_norm.weight": "indexer_compressor_norm.weight",
    "self_attn.compressor.indexer.q_b_proj.weight": "indexer.attn_q_b.weight",
    "self_attn.compressor.indexer.scorer.weights_proj.weight": "indexer.proj.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.gate.tid2eid": "ffn_gate_tid2eid.weight",
    "mlp.experts.down_proj": "ffn_down_exps.weight",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
}
LAYER = re.compile(r"layers\.(\d+)\.(.+)$")
# ggml integer types and their numpy element types.
INTEGERS = {"I8": "<i1", "I16": "<i2", "I32": "<i4", "I64": "<i8"}


class DeepseekV4Layout(ExpertLayout):
    """The converter names no tensor through TensorNameMap for this architecture, so alias maps nothing and compose
    gives every tensor: by TENSORS, the fused gate_up_proj from gate and up (ExpertLayout), and tid2eid as stored
    integers."""

    def __init__(self, gguf, config):
        super().__init__(gguf, config)
        hashed = [i for i, kind in enumerate(getattr(config, "mlp_layer_types", None) or ()) if kind == "hash_moe"]
        missing = [i for i in hashed if f"blk.{i}.ffn_gate_tid2eid.weight" not in gguf.tensors]
        if missing:
            raise GgufLayoutError(f"hash-routed layers {missing} have no ffn_gate_tid2eid table in the GGUF")

    def alias(self, name):
        return ""

    @staticmethod
    def gguf_name(name):
        """The GGUF tensor the converter writes for modeling-code tensor `name` (HF key), or None."""
        if name == "lm_head.weight":
            return "output.weight"
        name = name.removeprefix("model.")
        if name in TENSORS:
            return TENSORS[name]
        match = LAYER.fullmatch(name)
        if match and match[2] in LAYER_TENSORS:
            return f"blk.{match[1]}.{LAYER_TENSORS[match[2]]}"
        return None

    def compose(self, name, present):
        experts = super().compose(name, present)
        if experts is not None:
            return experts
        stored = self.gguf_name(name)
        if stored is None or stored not in present:
            return None
        if name.endswith(".tid2eid"):
            return (), lambda: self._integers(stored)
        return (stored,), lambda tensor: tensor

    def _integers(self, gguf_name):
        """An integer GGUF tensor as stored (torch, CPU, row-major)."""
        import numpy as np
        import torch
        tensor = self.gguf.tensors[gguf_name]
        if tensor.dtype not in INTEGERS:
            raise GgufLayoutError(f"{gguf_name}: {tensor.dtype}, expected an integer table")
        values = np.fromfile(tensor.shard, dtype=INTEGERS[tensor.dtype], count=tensor.elements,
                             offset=tensor.data_begin)
        return torch.from_numpy(values.reshape(tensor.shape).copy())
