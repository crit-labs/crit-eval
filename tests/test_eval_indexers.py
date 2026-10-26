"""crit_eval.indexers: the reference's faster forms of modeling code select what transformers' own loops select, and
its blocked DeepSeek V4 indexer scorer scores what transformers' scorer does."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import indexers  # noqa: E402

try:
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen4
except ImportError:
    torch = qwen4 = None
try:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as deepseek4
except ImportError:
    deepseek4 = None


class Keys:
    """The indexer's part of a cache: the keys of the chunks before, then this chunk's."""

    def __init__(self, past):
        self.past = past

    def update_indexer(self, keys, layer):
        return torch.cat([self.past, keys], dim=1)


@unittest.skipIf(qwen4 is None, "needs torch and transformers with qwen4_exp")
class QsaIndexerTests(unittest.TestCase):
    def setUp(self):
        # Sixteen heads: a block scores zero only when every head's ReLU is zero, so ties are rare here (the tie tests
        # build them with fewer heads).
        self.module = self.indexer(16)

    @staticmethod
    def indexer(heads, budget=24):
        torch.manual_seed(0)
        config = SimpleNamespace(hidden_size=64, indexer_n_heads=heads, indexer_kv_heads=1, indexer_head_dim=16,
                                 indexer_budget=budget, indexer_compress_ratio=4, rms_norm_eps=1e-6)
        module = qwen4.Qwen4ExpTextQSAIndexer(config, 0)   # budget / 4 blocks of four tokens per query
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(0, 0.5)
        return module

    def inputs(self, length, past=0, batch=2):
        """Hidden states of `length` queries after `past` keys, RoPE over 8 of the 16 dims, a causal bool mask."""
        hidden = torch.randn(batch, length, 64)
        kv = past + length
        angle = torch.arange(kv, dtype=torch.float32)[:, None] * 10000 ** (-torch.arange(4) / 4)[None]
        cos = torch.cat([angle.cos()] * 2, -1)[None].expand(batch, kv, 8).contiguous()
        sin = torch.cat([angle.sin()] * 2, -1)[None].expand(batch, kv, 8).contiguous()
        mask = (torch.arange(kv)[None] <= (past + torch.arange(length))[:, None])[None, None].expand(batch, 1, length, kv)
        cache = Keys(torch.randn(batch, past, 16)) if past else None
        return hidden, (cos, sin), mask.contiguous(), cache

    def same(self, *args, module=None):
        """The form's mask, checked equal to the loop's."""
        module = module or self.module
        with torch.no_grad():
            loop = type(module).forward(module, *args)
            fast = indexers.qsa_indexer_forward(module, *args)
        self.assertEqual(fast.dtype, loop.dtype)
        self.assertTrue(torch.equal(fast, loop))
        return fast

    @staticmethod
    def edge_ties(module, hidden, rope_pair, mask):
        """How many queries of the first row (no cache) have a tie at their top-k edge in the loop's own scores: the
        block they keep last and the first they drop score the same."""
        cos, sin = rope_pair
        rope = qwen4.apply_rotary_pos_emb
        with torch.no_grad():
            q, keys = module.index_qk_proj(hidden[:1]).split(
                [module.index_n_heads * module.index_head_dim, module.index_head_dim], dim=-1)
            q = rope(module.q_layernorm(q.reshape(1, -1, module.index_n_heads, module.index_head_dim)),
                     cos=cos[:1], sin=sin[:1], unsqueeze_dim=2)[0]
            ties = 0
            for query in range(hidden.shape[1]):
                n = int(mask[0, 0, query].sum()) // module.compress_ratio
                if n <= module.block_topk:
                    continue
                groups = keys[0, :n * module.compress_ratio].view(n, module.compress_ratio, -1)
                starts = torch.arange(n) * module.compress_ratio
                pooled = rope(module.k_layernorm(groups.float().mean(dim=1)).unsqueeze(1), cos=cos[0, starts],
                              sin=sin[0, starts]).squeeze(1)
                scores = torch.relu(q[query].float() @ pooled.float().T).sum(0).sort(descending=True).values
                ties += bool(scores[module.block_topk - 1] == scores[module.block_topk])
        return ties

    def test_the_selection_matches_the_loop_for_causal_prompts_and_resumed_chunks(self):
        for length, past in ((3, 0), (23, 0), (200, 0), (64, 150), (131, 37)):
            with self.subTest(length=length, past=past):
                mask = self.same(*self.inputs(length, past))
                # Far enough in, a query keeps its budget of blocks plus its tail, not everything it sees.
                if past + length > 40:
                    self.assertLess(int(mask[0, 0, -1].sum()), past + length)

    def test_ties_at_the_top_k_edge_select_what_the_loop_selects(self):
        # One or two heads: about half or a quarter of the blocks score exactly 0 after the ReLU, so far enough in a
        # query has fewer positive blocks than its budget and the loop's top-k picks among zeros.
        for heads in (1, 2):
            module = self.indexer(heads, budget=8)
            for length, past in ((23, 0), (200, 0), (131, 37)):
                with self.subTest(heads=heads, length=length, past=past):
                    inputs = self.inputs(length, past)
                    self.same(*inputs, module=module)
                    if not past:   # the ties are there (the cached keys of resumed chunks are not the projection's)
                        self.assertGreater(self.edge_ties(module, *inputs[:3]), 0)

    def test_equal_scores_select_what_the_loop_selects(self):
        # Every block the same tokens and RoPE the identity: all blocks score alike, a tie for every query that sees
        # more blocks than its budget (2 blocks; the 53 queries from the 12th token on see 3 or more).
        module = self.indexer(4, budget=8)
        hidden, (cos, sin), mask, _ = self.inputs(64)
        hidden, rope_pair = hidden[:, :4].repeat(1, 16, 1), (torch.ones_like(cos), torch.zeros_like(sin))
        self.same(hidden, rope_pair, mask, None, module=module)
        self.assertEqual(self.edge_ties(module, hidden, rope_pair, mask), 64 - 11)

    def test_a_float_mask_gives_the_loops_float_mask(self):
        hidden, rope, mask, cache = self.inputs(90, 20)
        additive = torch.where(mask, torch.zeros(()), torch.finfo(torch.float32).min)
        self.same(hidden, rope, additive, cache)

    def test_a_mask_that_is_no_prefix_runs_transformers_own_loop(self):
        hidden, rope, mask, cache = self.inputs(40)
        padded = mask.clone()
        padded[1, :, :, :5] = False   # left padding in the second row
        self.same(hidden, rope, padded, cache)

    def test_the_form_binds_only_to_the_source_it_was_checked_against(self):
        self.assertEqual(indexers.source_hash(qwen4.Qwen4ExpTextQSAIndexer.forward),
                         indexers.CHECKED["Qwen4ExpTextQSAIndexer"],
                         "transformers changed the indexer: check the form against it again and update CHECKED")
        holder = torch.nn.ModuleList([self.module])
        self.assertEqual(indexers.exact_forms(holder), ["Qwen4ExpTextQSAIndexer"])
        self.assertEqual(self.module.forward.__func__, indexers.qsa_indexer_forward)


@unittest.skipIf(qwen4 is None, "needs torch and transformers with qwen4_exp")
class QsaModelTests(unittest.TestCase):
    """The tiny model of the qwen4exp GGUF round trip (tests/test_eval_gguf_qwen4exp.py): on its 23-token text the
    form once kept other zero-scoring blocks than the loop, 0.43 nats off on a target."""

    def test_the_model_scores_alike_with_the_form_and_the_loop(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from test_eval_gguf_qwen4exp import TINY
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model("qwen4_exp_text", vocab_size=320, eos_token_id=0,
                                                                      **TINY), dtype=torch.float32)
        model.config._experts_implementation = "eager"
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.2 + (0.5 if "norm" in key else 0)
                                - (1.0 if "A_log" in key else 0))
        torch.manual_seed(1)
        texts = [torch.randint(0, 300, (40,)), torch.randint(0, 300, (23,))]
        masks = []
        indexer = model.model.layers[3].self_attn.indexer
        indexer.register_forward_hook(lambda module, args, output: masks.append(output))

        def logits():
            with torch.no_grad():
                return [model(ids[None], use_cache=False).logits[0] for ids in texts]
        loop = logits()
        self.assertEqual(indexers.exact_forms(model), ["Qwen4ExpTextQSAIndexer"])
        fast = logits()
        for a, b in zip(masks[:2], masks[2:]):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(loop, fast):
            self.assertTrue(torch.equal(a, b))


@unittest.skipIf(torch is None or deepseek4 is None, "needs torch and transformers with deepseek_v4")
class IndexerScorerTests(unittest.TestCase):
    """DeepseekV4IndexerScorer in blocks of query rows: the scores of transformers' forward bit for bit, on the CPU and,
    where there is one, on the GPU at the released model's index heads (64 of 128) and hidden size."""

    @staticmethod
    def scorer(heads, dim, hidden, device):
        torch.manual_seed(0)
        config = SimpleNamespace(index_head_dim=dim, index_n_heads=heads, hidden_size=hidden)
        module = deepseek4.DeepseekV4IndexerScorer(config).to(device)
        with torch.no_grad():
            module.weights_proj.weight.normal_(0, 0.05)
        return module

    def same(self, module, rows, keys, batch=1, limit=None):
        """The blocked scores, checked equal to transformers' (blocks of SCORER_BYTES `limit`)."""
        from unittest import mock
        device = module.weights_proj.weight.device
        q = torch.randn(batch, rows, module.weights_proj.out_features, int(module.softmax_scale ** -2), device=device)
        kv = torch.randn(batch, keys, q.shape[-1], device=device)
        hidden = torch.randn(batch, rows, module.weights_proj.in_features, device=device)
        with torch.no_grad(), mock.patch.object(indexers, "SCORER_BYTES", limit or indexers.SCORER_BYTES):
            whole = type(module).forward(module, q, kv, hidden)
            blocked = indexers.indexer_scorer_forward(module, q, kv, hidden)
        self.assertTrue(torch.equal(blocked, whole))

    def test_blocks_of_query_rows_score_as_the_whole_tensor(self):
        # Blocks of a few rows can round otherwise on the CPU (its batched matmul picks a kernel by the batch count:
        # 37 rows of 2 texts in blocks of 1 to 3); the form's blocks are SCORER_BYTES, hundreds of rows.
        module = self.scorer(4, 16, 64, "cpu")
        for rows, keys, batch, block in ((50, 13, 1, 7), (200, 40, 1, 16), (200, 40, 2, 33), (64, 30, 1, 64),
                                         (5, 0, 1, 2)):
            with self.subTest(rows=rows, keys=keys, batch=batch, block=block):
                self.same(module, rows, keys, batch, limit=max(1, block * batch * 4 * keys * 4))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs a CUDA device")
    def test_blocks_score_as_the_whole_tensor_on_the_gpu_at_the_models_shapes(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        module = self.scorer(64, 128, 4096, "cuda")
        for rows, keys in ((4096, 2048), (2048, 4096), (3000, 2047)):
            with self.subTest(rows=rows, keys=keys):
                self.same(module, rows, keys, limit=1 << 28)   # blocks of 512 to 2048 rows
        torch.cuda.empty_cache()

    def test_the_form_binds_only_to_the_source_it_was_checked_against(self):
        self.assertEqual(indexers.source_hash(deepseek4.DeepseekV4IndexerScorer.forward),
                         indexers.CHECKED["DeepseekV4IndexerScorer"],
                         "transformers changed the scorer: check the form against it again and update CHECKED")
        module = self.scorer(4, 16, 64, "cpu")
        self.assertEqual(indexers.exact_forms(torch.nn.ModuleList([module])), ["DeepseekV4IndexerScorer"])
        self.assertEqual(module.forward.__func__, indexers.indexer_scorer_forward)


if __name__ == "__main__":
    unittest.main()
