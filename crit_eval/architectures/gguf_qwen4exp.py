"""The reference converter's qwen4exp layout undone (Qwen3.8-Flash-Next; transformers' qwen4_exp_text), for
crit_eval.architectures.gguf_layout.

The converter (conversion/qwen4exp.py, Qwen4ExpTextModel) runs Qwen3.5's gated-delta-net steps (GatedDeltaNetLayout:
value heads tiled, A_log as -exp(A_log), dt_bias renamed, convolution kernels squeezed, zero-centred norms stored as
1 + w, experts stacked) and then:

    hyper-connections      renamed (hc_attn_*, hc_ffn_*, output_hc_*); their hc_norm gammas are zero-centred
                           (stored 1 + w by the norm rule above)
    QSA indexer            index_qk_proj split into indexer.q_proj (the query heads' rows) and indexer.k_proj (the
                           key head's); q_layernorm and k_layernorm stored 1 + w
    PLE                    norm_key, norm_query and norm_conv stored 1 + w, conv1d squeezed; the n-gram table,
                           stored in shards, concatenated into per_layer_token_embd; its int64 hash constants
                           (layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes) moved to metadata
                           (ple.layer_multipliers, .head_offsets, .head_vocab_sizes), since the converter casts
                           tensors that are not floating point to F32, which would round them
    shared-input merges    weights that read the same input written as one [a; b] (rows of a, then rows of b):
                           hc_attn_down_inject and hc_ffn_down_inject (input_mix_weight_down; block_inject_weight) on
                           every layer, attn_qkv_gate (in_proj_qkv; in_proj_z) and ssm_beta_alpha (in_proj_b;
                           in_proj_a) on linear-attention layers; converters before the merge (and GGUFs they wrote)
                           keep the halves, which read one to one
    MTP                    the draft block as nextn layer N (eh_proj from fc_embedding and fc_hidden): outside the text
                           model, never read

Undone here: each half of a merge is its rows of the merged tensor, index_qk_proj the query rows then the key rows,
the PLE norms w - 1 (one rounding, as the other norms), the n-gram table per_layer_token_embd as it is (its rows are
gathered, never the whole table decoded: GgufCheckpoint.row_source), and the hash constants the buffers read from
metadata exactly. Proven by the round trip in tests/test_eval_gguf_qwen4exp.py.
"""
import re

from .gguf_layout import GatedDeltaNetLayout, GgufLayoutError

BLOCK_NAME = re.compile(r"model\.layers\.(\d+)\.(.+)$")

# A PLE norm the converter stores 1 + w beyond those GatedDeltaNetLayout's norm rule (names ending norm.weight) covers.
PLE_NORMS = (".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight")
# The n-gram hash buffers -> the metadata key the converter moves each to.
PLE_CONSTANTS = {"layer_multipliers": ".ple.layer_multipliers", "ngram_heads_offsets": ".ple.head_offsets",
                 "ngram_heads_vocab_sizes": ".ple.head_vocab_sizes"}


class Qwen4ExpLayout(GatedDeltaNetLayout):
    """Qwen4ExpTextModel's tensors as transformers' Qwen4Exp names and lays them out (module docstring)."""

    def __init__(self, gguf, config):
        super().__init__(gguf, config)
        c = config
        self.ple_layers = [i - 1 for i in c.ple_layer_ids or ()]
        if len(self.ple_layers) > 1:
            raise GgufLayoutError(f"the config has {len(self.ple_layers)} PLE layers; a qwen4exp GGUF holds one n-gram "
                                  "table and one set of hash constants")
        # HF tensor in a layer -> (the merged GGUF tensor of its shared-input merge, which half it is, the first
        # half's rows).
        conv_dim = (2 * c.linear_num_key_heads * c.linear_key_head_dim
                    + c.linear_num_value_heads * c.linear_value_head_dim)
        self.merges = {"attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down_inject", 0, c.hc_lowrank),
                       "attn_hyper_connection.block_inject_weight.weight": ("hc_attn_down_inject", 1, c.hc_lowrank),
                       "mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down_inject", 0, c.hc_lowrank),
                       "mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_down_inject", 1, c.hc_lowrank),
                       "linear_attn.in_proj_qkv.weight": ("attn_qkv_gate", 0, conv_dim),
                       "linear_attn.in_proj_z.weight": ("attn_qkv_gate", 1, conv_dim),
                       "linear_attn.in_proj_b.weight": ("ssm_beta_alpha", 0, c.linear_num_value_heads),
                       "linear_attn.in_proj_a.weight": ("ssm_beta_alpha", 1, c.linear_num_value_heads)}
        self.query_rows = c.indexer_n_heads * c.indexer_head_dim

    def _ple_layer(self, name, part):
        """The layer of HF tensor `name` when it is `part` of the GGUF's PLE layer, else None."""
        for layer in self.ple_layers:
            if name == f"model.layers.{layer}.ple.{part}":
                return layer
        return None

    def alias(self, name):
        # The n-gram table maps as Gemma 3n's per-layer embedding table, the tensor the converter writes it to.
        if self._ple_layer(name, "ple_embedding.ngram_embedding.weight") is not None:
            return "model.embed_tokens_per_layer.weight"
        return super().alias(name)

    def inverse(self, name, tensor):
        if name.endswith(PLE_NORMS):
            return tensor - 1
        return super().inverse(name, tensor)

    def compose(self, name, present):
        composed = super().compose(name, present)
        if composed is not None:
            return composed
        found = BLOCK_NAME.match(name)
        if found is None:
            return None
        layer, part = int(found[1]), found[2]
        if part == "self_attn.indexer.index_qk_proj.weight":
            parts = (f"blk.{layer}.indexer.q_proj.weight", f"blk.{layer}.indexer.k_proj.weight")
            if not all(p in present for p in parts):
                return None
            import torch

            def joined(query, key):
                if query.shape[0] != self.query_rows:
                    raise GgufLayoutError(f"{parts[0]} has {query.shape[0]} rows; the config's indexer queries "
                                          f"{self.query_rows}")
                return torch.cat([query, key])
            return parts, joined
        if part in self.merges:
            merged, half, rows = self.merges[part]
            merged = f"blk.{layer}.{merged}.weight"
            if merged not in present:
                return None
            return (merged,), lambda tensor: tensor[:rows] if half == 0 else tensor[rows:]
        constant = part.removeprefix("ple.ple_embedding.")
        if constant in PLE_CONSTANTS and layer in self.ple_layers:
            values = self.gguf.get(PLE_CONSTANTS[constant])
            if not isinstance(values, tuple):
                return None
            import torch
            return (), lambda: torch.tensor(values, dtype=torch.int64)
        return None
