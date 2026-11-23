"""The reference's out-of-order forward sizes its free margin and its chunks from the device, not from one card."""
from types import SimpleNamespace
import unittest

try:
    from crit_eval.quality import reference
except ImportError:   # the reference's own dependencies (torch, transformers) are not installed here
    reference = None


@unittest.skipIf(reference is None, "the reference's dependencies")
class CalledChunkTests(unittest.TestCase):
    def chunk(self, heads, chunk, length, margin):
        model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=heads))
        return reference.StreamingReference._called_chunk(model, chunk, length, margin)

    def test_a_chunk_fits_its_per_chunk_tensors_in_the_margin(self):
        # 8 bytes a head, a chunk row and a position: 32 heads over 100,000 positions hold 167 rows in 4 GiB, 671 in 16
        self.assertEqual(self.chunk(32, 4096, 100_000, 4 << 30), 128)
        self.assertEqual(self.chunk(32, 4096, 100_000, 16 << 30), 512)
        self.assertEqual(self.chunk(32, 4096, 100_000, 1 << 40), 4096)    # never past the asked chunk
        self.assertEqual(self.chunk(32, 4096, 1000, 4 << 30), 4096)       # a short text keeps the asked chunk
        self.assertEqual(self.chunk(32, 4096, 100_000, 0), 16)            # at least 16 rows

    def test_the_margin_is_a_share_of_the_device(self):
        self.assertTrue(0 < reference.CALLED_FREE_SHARE < 1)


if __name__ == "__main__":
    unittest.main()
