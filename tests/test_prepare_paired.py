"""prepare renders a prompt from its messages as a server is sent them: each tool result paired with its call by id
(api_messages), so a template that names a tool result by its call renders the ids the server renders."""
import unittest

from crit_eval.quality.prepare import api_messages, paired, prompt_ids

MESSAGES = [{"role": "user", "content": "read it"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]},
            {"role": "tool", "content": "text"}]


class Recorder:
    def encode_chat(self, messages, **options):
        self.messages = messages
        return [1]


class PairedTests(unittest.TestCase):
    def test_tool_results_carry_their_calls_ids_as_the_server_is_sent_them(self):
        found, sent = paired(MESSAGES), api_messages(MESSAGES)
        self.assertEqual(found[1]["tool_calls"][0]["id"], sent[1]["tool_calls"][0]["id"])
        self.assertEqual(found[2]["tool_call_id"], sent[2]["tool_call_id"])
        self.assertEqual(found[1]["tool_calls"][0]["function"]["arguments"], {"path": "a"})   # objects, as rendered
        self.assertNotIn("tool_call_id", MESSAGES[2])

    def test_prompt_ids_render_the_paired_messages(self):
        model = Recorder()
        prompt_ids(model, MESSAGES)
        self.assertEqual(model.messages[2]["tool_call_id"], "call_0")


if __name__ == "__main__":
    unittest.main()
