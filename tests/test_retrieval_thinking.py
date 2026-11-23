"""Retrieval on a reasoning model (crit-eval-v0.8.1): asked with the protocol reasoning budget where the package ends
reasoning at one, judged on the answer; an engine without one is asked and judged as before."""
from types import SimpleNamespace
import unittest

from crit_eval.run.chat import answer_text, reply_text
from crit_eval.run.long_context import judged_retrieval, thinking_tokens

PREPARED = {"retrieval": {"token_text": [None, "ALPHA: 7", "thinking", " "], "facts": [{"id": "start", "answer": "ALPHA: 7"}]},
            "stop_ids": []}


class RetrievalThinkingTests(unittest.TestCase):
    def test_the_budget_only_where_the_package_ends_reasoning_at_one(self):
        rule = {"thinking_tokens": 128}
        self.assertEqual(thinking_tokens(SimpleNamespace(thinking_budget=True), rule), 128)
        self.assertIsNone(thinking_tokens(SimpleNamespace(thinking_budget=False), rule))
        self.assertIsNone(thinking_tokens(object(), rule))

    def test_a_reply_that_reasoned_is_judged_on_its_answer(self):
        verdict = judged_retrieval([2, 3, 1], PREPARED, [0] * 10, 4096, {"answer_tokens": [1]})
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["judged"], "answer")
        self.assertNotIn("judged", judged_retrieval([2, 3, 1], PREPARED, [0] * 10, 4096))

    def test_the_answer_leaves_the_reasoning_out(self):
        delta = {"role": "assistant", "reasoning_content": "hmm", "content": "ALPHA: 7"}
        self.assertEqual(answer_text(delta), "ALPHA: 7")
        self.assertEqual(reply_text(delta), "hmmALPHA: 7")


if __name__ == "__main__":
    unittest.main()
