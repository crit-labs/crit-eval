"""A speed case whose fixed conversation alone overflows its prompt budget under a model's template is not
measurable for that model: prepare records it, prefill is scored over the prefill cases left, the status says the
result is partial, and it does not compare with a result over another case set."""
import unittest

from crit_eval import prefill_cases
from crit_eval.quality import prepare
from crit_eval.reports import status


class UnmeasurableTests(unittest.TestCase):
    def test_prefill_cases_leave_out_what_the_data_could_not_build(self):
        protocol = {"speed": {"prefill_cases": ["long-summary", "agent-long"]}}
        self.assertEqual(prefill_cases(protocol), ["long-summary", "agent-long"])
        self.assertEqual(prefill_cases(protocol, {"agent-long": "overflows"}), ["long-summary"])

    def test_an_overflowing_frame_is_its_own_error(self):
        self.assertTrue(issubclass(prepare.FrameOverflow, ValueError))

    def test_results_over_other_case_sets_do_not_compare(self):
        full = {"model_sha256": "m", "suite_sha256": "s", "protocol": "crit-eval-v0.8.0"}
        partial = {**full, "speed_unmeasurable": {"agent-long": "the template renders 15,884 tokens of 14,336"}}
        found = status.changes(partial, full)
        self.assertFalse(found["comparable"])
        self.assertIn("agent-long", found["reason"])
        self.assertFalse(status.changes(full, partial)["comparable"])

    def test_the_protocol_states_the_rule(self):
        import json
        from pathlib import Path
        protocol = json.loads((Path(status.__file__).parents[1] / "protocol.json").read_text(encoding="utf-8"))
        self.assertIn("partial", protocol["speed"]["unmeasurable_cases"])


if __name__ == "__main__":
    unittest.main()
